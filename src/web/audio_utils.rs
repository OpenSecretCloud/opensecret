use serde::{Deserialize, Serialize};
use tracing::info;

const MAX_CHUNK_SIZE: usize = 20 * 1024 * 1024; // 20MB max chunk size
const CHUNK_OVERLAP_SECONDS: f64 = 1.0; // 1 second overlap between chunks
const MIN_WAV_HEADER_SIZE: usize = 44; // Minimum size for a valid WAV file header
pub const TINFOIL_MAX_SIZE: usize = 512 * 1024; // 0.5MB max size for Tinfoil provider

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioChunk {
    pub data: Vec<u8>,
    pub start_time: f64,
    pub end_time: f64,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub chunk_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedTranscription {
    pub text: String,
    pub segments: Option<Vec<TranscriptionSegment>>,
    pub language: Option<String>,
    pub duration: Option<f64>,
}

pub struct AudioSplitter {
    max_chunk_size: usize,
    overlap_seconds: f64,
}

impl Default for AudioSplitter {
    fn default() -> Self {
        Self {
            max_chunk_size: MAX_CHUNK_SIZE,
            overlap_seconds: CHUNK_OVERLAP_SECONDS,
        }
    }
}

impl AudioSplitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn should_split(&self, audio_data: &[u8]) -> bool {
        audio_data.len() > self.max_chunk_size
    }

    pub fn split_audio(
        &self,
        audio_data: &[u8],
        content_type: &str,
    ) -> Result<Vec<AudioChunk>, String> {
        if !self.should_split(audio_data) {
            // No need to split, return single chunk
            info!(
                "File size {} bytes is under limit, returning as single chunk",
                audio_data.len()
            );
            return Ok(vec![AudioChunk {
                data: audio_data.to_vec(),
                start_time: 0.0,
                end_time: 0.0,
                index: 0,
            }]);
        }

        info!(
            "File size {} bytes exceeds limit, splitting audio",
            audio_data.len()
        );

        // For WAV files, use simple WAV splitting
        // For other formats, we'll need to decode and re-encode
        match content_type {
            "audio/wav" | "audio/wave" => self.split_wav(audio_data),
            _ => {
                // For MP3 and other formats return full file as 1 chunk
                info!(
                    "Non-WAV format ({}), returning entire file as single chunk",
                    content_type
                );
                Ok(vec![AudioChunk {
                    data: audio_data.to_vec(),
                    start_time: 0.0,
                    end_time: 0.0, // Unknown duration
                    index: 0,
                }])
            }
        }
    }

    fn split_wav(&self, audio_data: &[u8]) -> Result<Vec<AudioChunk>, String> {
        // WAV file structure:
        // - RIFF header (12 bytes)
        // - fmt chunk (usually 24 bytes)
        // - data chunk header (8 bytes) + actual audio data

        if audio_data.len() < MIN_WAV_HEADER_SIZE {
            return Err(format!(
                "Invalid WAV file: too small (minimum {} bytes required)",
                MIN_WAV_HEADER_SIZE
            ));
        }

        // Verify it's a WAV file
        if &audio_data[0..4] != b"RIFF" || &audio_data[8..12] != b"WAVE" {
            return Err("Not a valid WAV file".to_string());
        }

        // Find both mandatory chunks. RIFF/WAVE requires fmt to precede data,
        // while optional and unknown chunks may appear around them.
        let mut pos = 12;
        let mut data_start = 0;
        let mut data_size = 0;
        let mut fmt_pos = 0;
        let mut fmt_size = 0;

        while pos + 8 <= audio_data.len() {
            let chunk_id = &audio_data[pos..pos + 4];
            let chunk_size = u32::from_le_bytes([
                audio_data[pos + 4],
                audio_data[pos + 5],
                audio_data[pos + 6],
                audio_data[pos + 7],
            ]) as usize;

            if chunk_id == b"fmt " {
                fmt_pos = pos + 8;
                fmt_size = chunk_size;
            } else if chunk_id == b"data" {
                if fmt_pos == 0 {
                    return Err("Invalid WAV file: fmt chunk must precede data chunk".to_string());
                }
                data_start = pos + 8;
                data_size = chunk_size;
            }

            // Stop if we've found both chunks
            if fmt_pos != 0 && data_start != 0 {
                break;
            }

            // Prevent integer overflow and infinite loops
            // Check if chunk_size would cause pos to overflow or exceed file bounds
            let next_pos = match pos.checked_add(8) {
                Some(p) => match p.checked_add(chunk_size) {
                    Some(next) => next,
                    None => {
                        return Err("Invalid WAV file: chunk size would cause overflow".to_string());
                    }
                },
                None => {
                    return Err("Invalid WAV file: position overflow".to_string());
                }
            };

            // Ensure we don't jump beyond the file bounds
            if next_pos > audio_data.len() {
                // Skip unknown chunks that extend beyond file bounds
                // This is common in truncated files
                break;
            }

            pos = next_pos;
            if chunk_size % 2 == 1 {
                // Check for overflow before adding padding byte
                pos = match pos.checked_add(1) {
                    Some(p) => p,
                    None => {
                        return Err("Invalid WAV file: position overflow with padding".to_string());
                    }
                };
            }
        }

        if fmt_pos == 0 {
            return Err("WAV file has no fmt chunk".to_string());
        }

        if data_start == 0 {
            return Err("WAV file has no data chunk".to_string());
        }

        // Validate fmt chunk size and bounds
        if fmt_size < 16 {
            return Err("WAV fmt chunk too small".to_string());
        }
        let fmt_end = fmt_pos
            .checked_add(fmt_size)
            .ok_or_else(|| "Invalid WAV file: fmt chunk size would cause overflow".to_string())?;
        if fmt_end > audio_data.len() {
            return Err("WAV fmt chunk exceeds file length".to_string());
        }

        // Validate that the data chunk size doesn't exceed the file length
        let data_end = data_start
            .checked_add(data_size)
            .ok_or_else(|| "Invalid WAV file: data chunk size would cause overflow".to_string())?;
        if data_end > audio_data.len() {
            return Err("WAV file data chunk exceeds file length".to_string());
        }

        // Extract header (everything before the data chunk header)
        if data_start < 8 {
            return Err("Invalid WAV file: data chunk offset too small".to_string());
        }
        let data_header_pos = data_start - 8;
        let header = &audio_data[0..data_header_pos];
        let audio_samples = &audio_data[data_start..data_end];

        // Parse format info from fmt chunk
        let fmt_data = &audio_data[fmt_pos..fmt_end];
        let channels = u16::from_le_bytes([fmt_data[2], fmt_data[3]]);
        let sample_rate = u32::from_le_bytes([fmt_data[4], fmt_data[5], fmt_data[6], fmt_data[7]]);
        let bytes_per_second =
            u32::from_le_bytes([fmt_data[8], fmt_data[9], fmt_data[10], fmt_data[11]]) as usize;
        let block_align = u16::from_le_bytes([fmt_data[12], fmt_data[13]]) as usize;

        // Validate format parameters to prevent division by zero
        if block_align == 0 {
            return Err("Invalid WAV file: block_align is zero".to_string());
        }
        if bytes_per_second == 0 {
            return Err("Invalid WAV file: bytes_per_second is zero".to_string());
        }

        info!(
            "WAV file: {} channels, {} Hz, {} bytes/sec",
            channels, sample_rate, bytes_per_second
        );

        // Calculate chunk sizes with checked math and alignment
        let header_overhead = header
            .len()
            .checked_add(8) // Header and data chunk header
            .ok_or_else(|| "WAV file header size would cause overflow".to_string())?;
        let available = self
            .max_chunk_size
            .checked_sub(header_overhead)
            .ok_or_else(|| {
                format!(
                    "WAV file header too large for chunking: {} bytes (max chunk size: {} bytes)",
                    header_overhead, self.max_chunk_size
                )
            })?;

        // Align to sample boundaries and ensure at least one frame
        let mut chunk_duration_bytes = (available / block_align)
            .checked_mul(block_align)
            .ok_or_else(|| "WAV sample-aligned chunk size would cause overflow".to_string())?;

        // RIFF chunks are word-aligned. Reserve a padding byte when a full
        // sample window would otherwise end at the size limit with odd length.
        let padded_chunk_duration_bytes = chunk_duration_bytes
            .checked_add(chunk_duration_bytes % 2)
            .ok_or_else(|| "WAV padded chunk size would cause overflow".to_string())?;
        if padded_chunk_duration_bytes > available {
            chunk_duration_bytes = chunk_duration_bytes
                .checked_sub(block_align)
                .ok_or_else(|| "WAV chunk size is smaller than one audio frame".to_string())?;
        }
        if chunk_duration_bytes < block_align {
            return Err(
                "Computed chunk size is smaller than one audio frame; increase max_chunk_size"
                    .to_string(),
            );
        }

        // Calculate overlap, ensuring it doesn't exceed chunk size
        let mut overlap_bytes =
            (((self.overlap_seconds * bytes_per_second as f64).round() as usize) / block_align)
                * block_align;
        if overlap_bytes >= chunk_duration_bytes {
            overlap_bytes = 0; // Disable overlap if it would exceed chunk size
        }

        let mut chunks = Vec::new();
        let mut offset = 0;
        let mut index = 0;

        while offset < audio_samples.len() {
            let start = if index > 0 && offset >= overlap_bytes {
                offset - overlap_bytes
            } else {
                offset
            };

            let end = start
                .saturating_add(chunk_duration_bytes)
                .min(audio_samples.len());

            // Ensure we're making progress to prevent infinite loops
            if end <= offset {
                return Err(
                    "Split produced a non-progressing window; check max_chunk_size/overlap"
                        .to_string(),
                );
            }

            let chunk_samples = &audio_samples[start..end];

            // Create a new WAV file for this chunk
            let chunk_wav = self.create_wav_from_samples(header, chunk_samples)?;

            let start_time = start as f64 / bytes_per_second as f64;
            let end_time = end as f64 / bytes_per_second as f64;

            chunks.push(AudioChunk {
                data: chunk_wav,
                start_time,
                end_time,
                index,
            });

            let chunk_size = chunks.last().map(|c| c.data.len()).unwrap_or(0);
            info!(
                "Created WAV chunk {} ({:.1}s - {:.1}s), size: {} bytes",
                index, start_time, end_time, chunk_size
            );

            offset = end;
            index += 1;
        }

        Ok(chunks)
    }

    fn create_wav_from_samples(
        &self,
        original_header: &[u8],
        samples: &[u8],
    ) -> Result<Vec<u8>, String> {
        let padding_size = samples.len() % 2;
        let output_size = original_header
            .len()
            .checked_add(8)
            .and_then(|size| size.checked_add(samples.len()))
            .and_then(|size| size.checked_add(padding_size))
            .ok_or_else(|| "WAV output size would cause overflow".to_string())?;
        if output_size > self.max_chunk_size {
            return Err(format!(
                "WAV output exceeds maximum chunk size: {output_size} bytes (max: {} bytes)",
                self.max_chunk_size
            ));
        }

        let data_size = u32::try_from(samples.len())
            .map_err(|_| "WAV data chunk exceeds the RIFF size limit".to_string())?;
        let riff_size = output_size
            .checked_sub(8)
            .and_then(|size| u32::try_from(size).ok())
            .ok_or_else(|| "WAV output exceeds the RIFF size limit".to_string())?;

        let mut wav = Vec::with_capacity(output_size);

        // Copy the original header up to the data chunk
        wav.extend_from_slice(original_header);

        // Update the data chunk size
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());

        // Add the actual audio samples
        wav.extend_from_slice(samples);
        if padding_size == 1 {
            wav.push(0);
        }

        // Update the RIFF chunk size (file size - 8)
        wav[4..8].copy_from_slice(&riff_size.to_le_bytes());

        Ok(wav)
    }
}

pub fn merge_transcriptions(
    results: Vec<(usize, serde_json::Value)>,
) -> Result<MergedTranscription, String> {
    if results.is_empty() {
        return Err("No transcription results to merge".to_string());
    }

    // Sort results by chunk index
    let mut sorted_results = results;
    sorted_results.sort_by_key(|(index, _)| *index);

    let mut merged_text = String::new();
    let mut all_segments = Vec::new();
    let mut detected_language: Option<String> = None;

    for (chunk_index, result) in sorted_results {
        // Extract text
        if let Some(text) = result.get("text").and_then(|t| t.as_str()) {
            if !merged_text.is_empty() {
                merged_text.push(' ');
            }
            merged_text.push_str(text);
        }

        // Extract language (use first detected)
        if detected_language.is_none() {
            if let Some(lang) = result.get("language").and_then(|l| l.as_str()) {
                detected_language = Some(lang.to_string());
            }
        }

        // Extract segments if available
        if let Some(segments) = result.get("segments").and_then(|s| s.as_array()) {
            for segment in segments {
                if let Some(seg_obj) = segment.as_object() {
                    all_segments.push(TranscriptionSegment {
                        text: seg_obj
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string(),
                        start: seg_obj.get("start").and_then(|s| s.as_f64()),
                        end: seg_obj.get("end").and_then(|e| e.as_f64()),
                        chunk_index,
                    });
                }
            }
        }
    }

    Ok(MergedTranscription {
        text: merged_text,
        segments: if all_segments.is_empty() {
            None
        } else {
            Some(all_segments)
        },
        language: detected_language,
        duration: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PCM_U8_MONO_FMT: [u8; 16] = [
        1, 0, // PCM
        1, 0, // mono
        0x40, 0x1f, 0, 0, // 8 kHz
        0x40, 0x1f, 0, 0, // 8,000 bytes/second
        1, 0, // one-byte frames
        8, 0, // eight bits/sample
    ];

    fn wav_with_chunks(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let mut wav = Vec::from(&b"RIFF\0\0\0\0WAVE"[..]);

        for (id, data) in chunks {
            wav.extend_from_slice(*id);
            wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
            wav.extend_from_slice(data);
            if data.len() % 2 == 1 {
                wav.push(0);
            }
        }

        let riff_size = (wav.len() - 8) as u32;
        wav[4..8].copy_from_slice(&riff_size.to_le_bytes());
        wav
    }

    fn parse_wav_chunks(wav: &[u8]) -> Vec<([u8; 4], Vec<u8>)> {
        assert!(wav.len() >= 12);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");

        let riff_size = u32::from_le_bytes(wav[4..8].try_into().unwrap()) as usize;
        assert_eq!(riff_size.checked_add(8), Some(wav.len()));

        let mut chunks = Vec::new();
        let mut pos = 12;
        let mut saw_fmt = false;
        let mut saw_data = false;

        while pos < wav.len() {
            let header_end = pos.checked_add(8).unwrap();
            assert!(header_end <= wav.len());

            let id: [u8; 4] = wav[pos..pos + 4].try_into().unwrap();
            let size = u32::from_le_bytes(wav[pos + 4..header_end].try_into().unwrap()) as usize;
            let data_end = header_end.checked_add(size).unwrap();
            assert!(data_end <= wav.len());

            if &id == b"fmt " {
                saw_fmt = true;
            } else if &id == b"data" {
                assert!(saw_fmt, "fmt chunk must precede data chunk");
                saw_data = true;
            }

            chunks.push((id, wav[header_end..data_end].to_vec()));
            pos = data_end.checked_add(size % 2).unwrap();
            assert!(pos <= wav.len());
            if size % 2 == 1 {
                assert_eq!(wav[pos - 1], 0);
            }
        }

        assert!(saw_fmt);
        assert!(saw_data);
        chunks
    }

    #[test]
    fn test_should_split() {
        let splitter = AudioSplitter::new();

        // Test data smaller than limit
        let small_data = vec![0u8; MAX_CHUNK_SIZE / 2];
        assert!(!splitter.should_split(&small_data));

        // Test data exactly at limit
        let exact_data = vec![0u8; MAX_CHUNK_SIZE];
        assert!(!splitter.should_split(&exact_data));

        // Test data just over limit
        let just_over_data = vec![0u8; MAX_CHUNK_SIZE + 1];
        assert!(splitter.should_split(&just_over_data));

        // Test data significantly over limit
        let large_data = vec![0u8; MAX_CHUNK_SIZE * 2];
        assert!(splitter.should_split(&large_data));
    }

    #[test]
    fn test_split_audio_small_file() {
        let splitter = AudioSplitter::new();

        // Test with small file that doesn't need splitting
        let small_audio = vec![0u8; 1024];
        let result = splitter.split_audio(&small_audio, "audio/mp3").unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].index, 0);
        assert_eq!(result[0].data.len(), 1024);
    }

    #[test]
    fn test_split_audio_non_wav_format() {
        let splitter = AudioSplitter::new();

        // Test with large MP3 file - should return as single chunk
        let large_mp3 = vec![0u8; MAX_CHUNK_SIZE * 2];
        let result = splitter.split_audio(&large_mp3, "audio/mp3").unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].data.len(), MAX_CHUNK_SIZE * 2);
    }

    #[test]
    fn test_split_wav_invalid_file() {
        let splitter = AudioSplitter::new();

        // Test with file too small to be valid WAV
        let tiny_file = vec![0u8; 10];
        let result = splitter.split_wav(&tiny_file);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too small"));

        // Test with file that's large enough but not a WAV
        let not_wav = vec![0u8; 100];
        let result = splitter.split_wav(&not_wav);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Not a valid WAV"));
    }

    #[test]
    fn test_split_wav_rejects_data_before_fmt() {
        let splitter = AudioSplitter::new();
        let wav = wav_with_chunks(&[(b"data", &[1, 2, 3, 4]), (b"fmt ", &PCM_U8_MONO_FMT)]);

        assert_eq!(
            splitter.split_wav(&wav).unwrap_err(),
            "Invalid WAV file: fmt chunk must precede data chunk"
        );
    }

    #[test]
    fn test_split_wav_emits_valid_word_aligned_chunks_with_optional_chunks() {
        let splitter = AudioSplitter {
            max_chunk_size: 71,
            overlap_seconds: 0.0,
        };
        let samples = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let wav = wav_with_chunks(&[
            (b"JUNK", &[10, 11, 12]),
            (b"fmt ", &PCM_U8_MONO_FMT),
            (b"LIST", &[13]),
            (b"data", &samples),
        ]);

        let chunks = splitter.split_wav(&wav).unwrap();
        assert!(chunks.len() > 1);

        let mut emitted_samples = Vec::new();
        for chunk in chunks {
            assert!(chunk.data.len() <= splitter.max_chunk_size);
            let parsed = parse_wav_chunks(&chunk.data);
            assert_eq!(
                parsed.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                vec![*b"JUNK", *b"fmt ", *b"LIST", *b"data"]
            );
            assert_eq!(parsed[0].1, [10, 11, 12]);
            assert_eq!(parsed[1].1, PCM_U8_MONO_FMT);
            assert_eq!(parsed[2].1, [13]);
            emitted_samples.extend_from_slice(&parsed[3].1);
        }

        assert_eq!(emitted_samples, samples);
    }

    #[test]
    fn test_merge_transcriptions() {
        let results = vec![
            (
                0,
                serde_json::json!({
                    "text": "Hello world",
                    "language": "en"
                }),
            ),
            (
                1,
                serde_json::json!({
                    "text": "this is a test",
                }),
            ),
            (
                2,
                serde_json::json!({
                    "text": "of merging transcriptions",
                }),
            ),
        ];

        let merged = merge_transcriptions(results).unwrap();
        assert_eq!(
            merged.text,
            "Hello world this is a test of merging transcriptions"
        );
        assert_eq!(merged.language, Some("en".to_string()));
    }
}
