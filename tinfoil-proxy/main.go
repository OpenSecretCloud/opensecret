package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/tinfoilsh/tinfoil-go"
)

// Model configurations
var modelConfigs = map[string]struct {
	ModelID     string
	Description string
	Active      bool
}{
	"deepseek-r1-0528": {
		ModelID:     "deepseek-r1-0528",
		Description: "Advanced reasoning and complex problem-solving model",
		Active:      true,
	},
	"deepseek-v31-terminus": {
		ModelID:     "deepseek-v31-terminus",
		Description: "Advanced reasoning and complex problem-solving model (V3.1 Terminus)",
		Active:      true,
	},
	"llama3-3-70b": {
		ModelID:     "llama3-3-70b",
		Description: "Multilingual understanding, dialogue optimization",
		Active:      true,
	},
	"nomic-embed-text": {
		ModelID:     "nomic-embed-text",
		Description: "Text embedding model",
		Active:      false,
	},
	"gpt-oss-120b": {
		ModelID:     "gpt-oss-120b",
		Description: "Powerful reasoning, configurable reasoning effort levels, full chain-of-thought access, native agentic abilities",
		Active:      true,
	},
	"qwen3-coder-480b": {
		ModelID:     "qwen3-coder-480b",
		Description: "Advanced coding model with 480B parameters for complex programming tasks",
		Active:      true,
	},
	"qwen3-vl-30b": {
		ModelID:     "qwen3-vl-30b",
		Description: "A 30B-parameter vision-language model that understands images and videos. Excels at visual tasks including GUI interaction, generating code from screenshots, spatial understanding, video analysis, and OCR across 32 languages. Supports up to 256K context for processing long videos and documents.",
		Active:      true,
	},
	"whisper-large-v3-turbo": {
		ModelID:     "whisper-large-v3-turbo",
		Description: "Fast and accurate speech-to-text transcription model",
		Active:      true,
	},
}

// Request/Response models
type ChatMessage struct {
	Role         string           `json:"role,omitempty"`
	Content      interface{}      `json:"content,omitempty"`
	ToolCalls    []map[string]any `json:"tool_calls,omitempty"`
	Refusal      string           `json:"refusal,omitempty"`
	FunctionCall map[string]any   `json:"function_call,omitempty"` // Deprecated but keeping for compatibility
}

type ChatCompletionRequest struct {
	Model             string          `json:"model"`
	Messages          []ChatMessage   `json:"messages"`
	Stream            *bool           `json:"stream,omitempty"`
	Temperature       *float32        `json:"temperature,omitempty"`
	MaxTokens         *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int          `json:"max_completion_tokens,omitempty"`
	TopP              *float32        `json:"top_p,omitempty"`
	FrequencyPenalty  *float32        `json:"frequency_penalty,omitempty"`
	PresencePenalty   *float32        `json:"presence_penalty,omitempty"`
	N                 *int            `json:"n,omitempty"`
	Stop              []string        `json:"stop,omitempty"`
	StreamOptions     *map[string]any `json:"stream_options,omitempty"`
	// Function calling / Tools
	Tools             []map[string]any `json:"tools,omitempty"`
	ToolChoice        any              `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitempty"`
	// Response formatting
	ResponseFormat    *map[string]any `json:"response_format,omitempty"`
	// Reasoning and verbosity controls
	ReasoningEffort   *string         `json:"reasoning_effort,omitempty"`
	Verbosity         *string         `json:"verbosity,omitempty"`
	// Log probabilities
	Logprobs          *bool           `json:"logprobs,omitempty"`
	TopLogprobs       *int            `json:"top_logprobs,omitempty"`
	// Determinism and storage
	Seed              *int            `json:"seed,omitempty"`
	Store             *bool           `json:"store,omitempty"`
	// User identification and caching
	User              *string         `json:"user,omitempty"`
	PromptCacheKey    *string         `json:"prompt_cache_key,omitempty"`
	SafetyIdentifier  *string         `json:"safety_identifier,omitempty"`
	// Advanced parameters
	LogitBias         map[string]int  `json:"logit_bias,omitempty"`
	Metadata          map[string]any  `json:"metadata,omitempty"`
	Modalities        []string        `json:"modalities,omitempty"`
	ServiceTier       *string         `json:"service_tier,omitempty"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

type Choice struct {
	Index        int          `json:"index"`
	Message      *ChatMessage `json:"message,omitempty"`
	Delta        *ChatMessage `json:"delta,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

type TTSRequest struct {
	Model          string  `json:"model"`
	Input          string  `json:"input"`
	Voice          string  `json:"voice"`
	ResponseFormat string  `json:"response_format,omitempty"`
	Speed          float32 `json:"speed,omitempty"`
	StreamFormat   string  `json:"stream_format,omitempty"`
}

type TranscriptionResponse struct {
	Text string `json:"text"`
}

type TinfoilProxyServer struct {
	client       *tinfoil.Client
	clientMutex  sync.RWMutex
	apiKey       string
	lastRotation time.Time
}

func NewTinfoilProxyServer() (*TinfoilProxyServer, error) {
	apiKey := os.Getenv("TINFOIL_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("TINFOIL_API_KEY environment variable is required")
	}

	server := &TinfoilProxyServer{
		apiKey: apiKey,
	}

	// Initialize Tinfoil client with new simplified API
	log.Printf("Initializing Tinfoil client with new API")

	// Create a single client that will handle all models through the inference endpoint
	client, err := tinfoil.NewClient(option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Tinfoil client: %v", err)
	}

	server.client = client
	server.lastRotation = time.Now()
	log.Printf("Successfully initialized Tinfoil client")

	// Start automatic client rotation every 10 minutes
	go server.startAutoRotation()

	return server, nil
}

func (s *TinfoilProxyServer) getClient() (*tinfoil.Client, error) {
	s.clientMutex.RLock()
	defer s.clientMutex.RUnlock()

	if s.client == nil {
		return nil, fmt.Errorf("client not initialized")
	}
	return s.client, nil
}

// startAutoRotation rotates the client every 10 minutes
func (s *TinfoilProxyServer) startAutoRotation() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		log.Printf("Performing scheduled client rotation")
		if err := s.reinitializeClient(); err != nil {
			log.Printf("Failed to rotate client on schedule: %v", err)
		}
	}
}

// reinitializeClient creates a new client instance, typically called after certificate errors
func (s *TinfoilProxyServer) reinitializeClient() error {
	// Check if we recently rotated to avoid excessive reinitializations
	s.clientMutex.RLock()
	timeSinceLastRotation := time.Since(s.lastRotation)
	s.clientMutex.RUnlock()

	if timeSinceLastRotation < 30*time.Second {
		log.Printf("Skipping reinitialization - client was rotated %.0f seconds ago", timeSinceLastRotation.Seconds())
		return nil
	}

	log.Printf("Reinitializing Tinfoil client (last rotation: %.0f seconds ago)", timeSinceLastRotation.Seconds())

	client, err := tinfoil.NewClient(option.WithAPIKey(s.apiKey))
	if err != nil {
		return fmt.Errorf("failed to reinitialize Tinfoil client: %v", err)
	}

	s.clientMutex.Lock()
	s.client = client
	s.lastRotation = time.Now()
	s.clientMutex.Unlock()

	log.Printf("Successfully reinitialized Tinfoil client")
	return nil
}

// isCertificateError checks if an error is related to certificate issues
func isCertificateError(err error) bool {
	if err == nil {
		return false
	}
	errStr := err.Error()
	return strings.Contains(errStr, "certificate") ||
		strings.Contains(errStr, "fingerprint") ||
		strings.Contains(errStr, "x509") ||
		strings.Contains(errStr, "tls")
}

// convertToOpenAIMessage handles both string content and multimodal content arrays
func convertToOpenAIMessage(msg ChatMessage, role string) openai.ChatCompletionMessageParamUnion {
	// If content is a string, use the simple message constructors
	if contentStr, ok := msg.Content.(string); ok {
		switch role {
		case "user":
			return openai.UserMessage(contentStr)
		case "assistant":
			return openai.AssistantMessage(contentStr)
		case "system":
			return openai.SystemMessage(contentStr)
		default:
			return openai.UserMessage(contentStr)
		}
	}

	// If content is an array, it's multimodal content
	if contentArray, ok := msg.Content.([]interface{}); ok {
		var parts []openai.ChatCompletionContentPartUnionParam

		for _, part := range contentArray {
			if partMap, ok := part.(map[string]interface{}); ok {
				if partType, exists := partMap["type"].(string); exists {
					switch partType {
					case "text":
						if text, ok := partMap["text"].(string); ok {
							parts = append(parts, openai.TextContentPart(text))
						}
					case "image_url":
						if imageURLMap, ok := partMap["image_url"].(map[string]interface{}); ok {
							if url, ok := imageURLMap["url"].(string); ok {
								parts = append(parts, openai.ImageContentPart(
									openai.ChatCompletionContentPartImageImageURLParam{
										URL: url,
									},
								))
							}
						}
					}
				}
			}
		}

		// Only user messages support multimodal content in the OpenAI SDK
		if role == "user" && len(parts) > 0 {
			return openai.UserMessage(parts)
		}
	}

	// Fallback: stringify the content
	contentJSON, _ := json.Marshal(msg.Content)
	switch role {
	case "user":
		return openai.UserMessage(string(contentJSON))
	case "assistant":
		return openai.AssistantMessage(string(contentJSON))
	case "system":
		return openai.SystemMessage(string(contentJSON))
	default:
		return openai.UserMessage(string(contentJSON))
	}
}

func (s *TinfoilProxyServer) streamChatCompletion(c *gin.Context, req ChatCompletionRequest) {
	client, err := s.getClient()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Convert messages to OpenAI format
	messages := make([]openai.ChatCompletionMessageParamUnion, len(req.Messages))
	for i, msg := range req.Messages {
		switch msg.Role {
		case "user":
			messages[i] = convertToOpenAIMessage(msg, "user")
		case "assistant":
			messages[i] = convertToOpenAIMessage(msg, "assistant")
		case "system":
			messages[i] = convertToOpenAIMessage(msg, "system")
		default:
			messages[i] = convertToOpenAIMessage(msg, "user")
		}
	}

	// Build chat completion params
	params := openai.ChatCompletionNewParams{
		Model:    req.Model,
		Messages: messages,
	}

	// Add optional parameters if provided
	if req.Temperature != nil {
		params.Temperature = openai.Float(float64(*req.Temperature))
	}
	if req.MaxTokens != nil {
		params.MaxTokens = openai.Int(int64(*req.MaxTokens))
	}
	if req.TopP != nil {
		params.TopP = openai.Float(float64(*req.TopP))
	}
	if req.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(float64(*req.FrequencyPenalty))
	}
	if req.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(float64(*req.PresencePenalty))
	}
	if req.N != nil {
		params.N = openai.Int(int64(*req.N))
	}
	// Note: Stop parameter handling is complex in the OpenAI Go SDK v1.3.0
	// For now, we'll skip this parameter
	if req.StreamOptions != nil {
		// Pass stream options to the params
		params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool((*req.StreamOptions)["include_usage"] == true),
		}
	}

	// Add tools support (function calling)
	if req.Tools != nil && len(req.Tools) > 0 {
		// Convert tools from raw JSON to SDK types
		toolsJSON, _ := json.Marshal(req.Tools)
		var tools []openai.ChatCompletionToolUnionParam
		if err := json.Unmarshal(toolsJSON, &tools); err != nil {
			log.Printf("Failed to unmarshal tools: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid tools format"})
			return
		}
		params.Tools = tools
	}
	if req.ToolChoice != nil {
		// Convert tool_choice from raw JSON to SDK type
		toolChoiceJSON, _ := json.Marshal(req.ToolChoice)
		var toolChoice openai.ChatCompletionToolChoiceOptionUnionParam
		if err := json.Unmarshal(toolChoiceJSON, &toolChoice); err != nil {
			log.Printf("Failed to unmarshal tool_choice: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid tool_choice format"})
			return
		}
		params.ToolChoice = toolChoice
	}
	if req.ParallelToolCalls != nil {
		params.ParallelToolCalls = openai.Bool(*req.ParallelToolCalls)
	}
	if req.ResponseFormat != nil {
		// Convert response_format from raw JSON to SDK type
		responseFormatJSON, _ := json.Marshal(req.ResponseFormat)
		var responseFormat openai.ChatCompletionNewParamsResponseFormatUnion
		if err := json.Unmarshal(responseFormatJSON, &responseFormat); err != nil {
			log.Printf("Failed to unmarshal response_format: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid response_format"})
			return
		}
		params.ResponseFormat = responseFormat
	}

	// Add new parameters
	if req.MaxCompletionTokens != nil {
		params.MaxCompletionTokens = openai.Int(int64(*req.MaxCompletionTokens))
	}
	if req.ReasoningEffort != nil {
		params.ReasoningEffort = openai.ReasoningEffort(*req.ReasoningEffort)
	}
	if req.Verbosity != nil {
		params.Verbosity = openai.ChatCompletionNewParamsVerbosity(*req.Verbosity)
	}
	if req.Logprobs != nil {
		params.Logprobs = openai.Bool(*req.Logprobs)
	}
	if req.TopLogprobs != nil {
		params.TopLogprobs = openai.Int(int64(*req.TopLogprobs))
	}
	if req.Seed != nil {
		params.Seed = openai.Int(int64(*req.Seed))
	}
	if req.Store != nil {
		params.Store = openai.Bool(*req.Store)
	}
	if req.User != nil {
		params.User = openai.String(*req.User)
	}
	if req.PromptCacheKey != nil {
		params.PromptCacheKey = openai.String(*req.PromptCacheKey)
	}
	if req.SafetyIdentifier != nil {
		params.SafetyIdentifier = openai.String(*req.SafetyIdentifier)
	}
	if req.LogitBias != nil && len(req.LogitBias) > 0 {
		logitBias := make(map[string]int64)
		for k, v := range req.LogitBias {
			logitBias[k] = int64(v)
		}
		params.LogitBias = logitBias
	}
	if req.Metadata != nil && len(req.Metadata) > 0 {
		metadataJSON, _ := json.Marshal(req.Metadata)
		var metadata map[string]string
		if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
			log.Printf("Failed to unmarshal metadata: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid metadata format"})
			return
		}
		params.Metadata = metadata
	}
	if req.Modalities != nil && len(req.Modalities) > 0 {
		params.Modalities = req.Modalities
	}
	if req.ServiceTier != nil {
		params.ServiceTier = openai.ChatCompletionNewParamsServiceTier(*req.ServiceTier)
	}

	// Create context for cancellation
	ctx := c.Request.Context()

	// Start streaming - but first check if we can create the stream without errors
	stream := client.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()

	// Try to get the first chunk to detect early errors before sending SSE headers
	if !stream.Next() {
		if err := stream.Err(); err != nil {
			log.Printf("Stream creation error: %v", err)

			// Check if this is a certificate error and reinitialize if needed
			if isCertificateError(err) {
				go func() {
					if reinitErr := s.reinitializeClient(); reinitErr != nil {
						log.Printf("Failed to reinitialize client: %v", reinitErr)
					}
				}()
			}

			// Return proper HTTP error before SSE headers are sent
			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error": map[string]interface{}{
					"message": "Failed to connect to upstream service",
					"type":    "server_error",
					"code":    "upstream_error",
				},
			})
			return
		}
		// Stream ended without error but also without data
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": map[string]interface{}{
				"message": "No response from upstream service",
				"type":    "server_error",
				"code":    "empty_stream",
			},
		})
		return
	}

	// We have at least one chunk, so we can proceed with SSE
	// Set up SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	// Stream responses
	w := c.Writer
	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Printf("Response writer does not support flushing")
		return
	}

	// Track if we've already sent usage data for this stream
	usageSent := false

	// Process the first chunk we already read
	firstChunk := true

	for firstChunk || stream.Next() {
		if firstChunk {
			firstChunk = false
		}
		chunk := stream.Current()

		// Convert to OpenAI-compatible format
		chunkData := ChatCompletionResponse{
			ID:      chunk.ID,
			Object:  "chat.completion.chunk",
			Created: chunk.Created,
			Model:   req.Model,
			Choices: make([]Choice, 0),
		}

		// Track if this chunk has a finish_reason
		hasFinishReason := false

		// Handle empty choices array - this appears to be Tinfoil's way of signaling end
		if len(chunk.Choices) == 0 {
			// Inject a proper final chunk with finish_reason
			log.Printf("Empty choices array detected - injecting finish_reason: 'stop'")
			finishReason := "stop"
			choiceData := Choice{
				Index:        0,
				Delta:        &ChatMessage{},
				FinishReason: &finishReason,
			}
			chunkData.Choices = append(chunkData.Choices, choiceData)
			hasFinishReason = true
		} else {
			// Use marshal/unmarshal to preserve all fields including tool_calls
			for _, choice := range chunk.Choices {
				// Marshal the SDK choice to JSON
				choiceJSON, err := json.Marshal(choice)
				if err != nil {
					log.Printf("Failed to marshal choice: %v", err)
					continue
				}

				// Unmarshal into our Choice type to preserve all fields
				var choiceData Choice
				if err := json.Unmarshal(choiceJSON, &choiceData); err != nil {
					log.Printf("Failed to unmarshal choice: %v", err)
					continue
				}

				// Check if this chunk has a finish_reason
				if choiceData.FinishReason != nil && *choiceData.FinishReason != "" {
					hasFinishReason = true
				}

				chunkData.Choices = append(chunkData.Choices, choiceData)
			}
		}

		// Include usage data only on final chunk (when we have a finish_reason AND completion tokens)
		// This prevents sending usage data on intermediate chunks or chunks with only prompt tokens
		// Also ensure we only send usage once per stream
		if chunk.Usage.TotalTokens > 0 && chunk.Usage.CompletionTokens > 0 && hasFinishReason && !usageSent {
			chunkData.Usage = &Usage{
				PromptTokens:     int(chunk.Usage.PromptTokens),
				CompletionTokens: int(chunk.Usage.CompletionTokens),
				TotalTokens:      int(chunk.Usage.TotalTokens),
			}
			usageSent = true
		}

		data, err := json.Marshal(chunkData)
		if err != nil {
			log.Printf("Failed to marshal chunk data: %v", err)
			// Just terminate the stream cleanly without exposing the error
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	if err := stream.Err(); err != nil {
		log.Printf("Stream error: %v", err)

		// Check if this is a certificate error and reinitialize if needed
		if isCertificateError(err) {
			go func() {
				if reinitErr := s.reinitializeClient(); reinitErr != nil {
					log.Printf("Failed to reinitialize client: %v", reinitErr)
				}
			}()
		}

		// Send error to client in OpenAI-compatible format
		errorResponse := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "Stream processing error",
				"type":    "server_error",
				"code":    "stream_error",
			},
		}
		data, _ := json.Marshal(errorResponse)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		return
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *TinfoilProxyServer) nonStreamingChatCompletion(c *gin.Context, req ChatCompletionRequest) {
	client, err := s.getClient()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Convert messages to OpenAI format
	messages := make([]openai.ChatCompletionMessageParamUnion, len(req.Messages))
	for i, msg := range req.Messages {
		switch msg.Role {
		case "user":
			messages[i] = convertToOpenAIMessage(msg, "user")
		case "assistant":
			messages[i] = convertToOpenAIMessage(msg, "assistant")
		case "system":
			messages[i] = convertToOpenAIMessage(msg, "system")
		default:
			messages[i] = convertToOpenAIMessage(msg, "user")
		}
	}

	// Build chat completion params
	params := openai.ChatCompletionNewParams{
		Model:    req.Model,
		Messages: messages,
	}

	// Add optional parameters if provided (same as streaming)
	if req.Temperature != nil {
		params.Temperature = openai.Float(float64(*req.Temperature))
	}
	if req.MaxTokens != nil {
		params.MaxTokens = openai.Int(int64(*req.MaxTokens))
	}
	if req.TopP != nil {
		params.TopP = openai.Float(float64(*req.TopP))
	}
	if req.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(float64(*req.FrequencyPenalty))
	}
	if req.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(float64(*req.PresencePenalty))
	}
	if req.N != nil {
		params.N = openai.Int(int64(*req.N))
	}

	// Add tools support (function calling)
	if req.Tools != nil && len(req.Tools) > 0 {
		// Convert tools from raw JSON to SDK types
		toolsJSON, _ := json.Marshal(req.Tools)
		var tools []openai.ChatCompletionToolUnionParam
		if err := json.Unmarshal(toolsJSON, &tools); err != nil {
			log.Printf("Failed to unmarshal tools: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid tools format"})
			return
		}
		params.Tools = tools
	}
	if req.ToolChoice != nil {
		// Convert tool_choice from raw JSON to SDK type
		toolChoiceJSON, _ := json.Marshal(req.ToolChoice)
		var toolChoice openai.ChatCompletionToolChoiceOptionUnionParam
		if err := json.Unmarshal(toolChoiceJSON, &toolChoice); err != nil {
			log.Printf("Failed to unmarshal tool_choice: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid tool_choice format"})
			return
		}
		params.ToolChoice = toolChoice
	}
	if req.ParallelToolCalls != nil {
		params.ParallelToolCalls = openai.Bool(*req.ParallelToolCalls)
	}
	if req.ResponseFormat != nil {
		// Convert response_format from raw JSON to SDK type
		responseFormatJSON, _ := json.Marshal(req.ResponseFormat)
		var responseFormat openai.ChatCompletionNewParamsResponseFormatUnion
		if err := json.Unmarshal(responseFormatJSON, &responseFormat); err != nil {
			log.Printf("Failed to unmarshal response_format: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid response_format"})
			return
		}
		params.ResponseFormat = responseFormat
	}

	// Add new parameters (same as streaming)
	if req.MaxCompletionTokens != nil {
		params.MaxCompletionTokens = openai.Int(int64(*req.MaxCompletionTokens))
	}
	if req.ReasoningEffort != nil {
		params.ReasoningEffort = openai.ReasoningEffort(*req.ReasoningEffort)
	}
	if req.Verbosity != nil {
		params.Verbosity = openai.ChatCompletionNewParamsVerbosity(*req.Verbosity)
	}
	if req.Logprobs != nil {
		params.Logprobs = openai.Bool(*req.Logprobs)
	}
	if req.TopLogprobs != nil {
		params.TopLogprobs = openai.Int(int64(*req.TopLogprobs))
	}
	if req.Seed != nil {
		params.Seed = openai.Int(int64(*req.Seed))
	}
	if req.Store != nil {
		params.Store = openai.Bool(*req.Store)
	}
	if req.User != nil {
		params.User = openai.String(*req.User)
	}
	if req.PromptCacheKey != nil {
		params.PromptCacheKey = openai.String(*req.PromptCacheKey)
	}
	if req.SafetyIdentifier != nil {
		params.SafetyIdentifier = openai.String(*req.SafetyIdentifier)
	}
	if req.LogitBias != nil && len(req.LogitBias) > 0 {
		logitBias := make(map[string]int64)
		for k, v := range req.LogitBias {
			logitBias[k] = int64(v)
		}
		params.LogitBias = logitBias
	}
	if req.Metadata != nil && len(req.Metadata) > 0 {
		metadataJSON, _ := json.Marshal(req.Metadata)
		var metadata map[string]string
		if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
			log.Printf("Failed to unmarshal metadata: %v", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid metadata format"})
			return
		}
		params.Metadata = metadata
	}
	if req.Modalities != nil && len(req.Modalities) > 0 {
		params.Modalities = req.Modalities
	}
	if req.ServiceTier != nil {
		params.ServiceTier = openai.ChatCompletionNewParamsServiceTier(*req.ServiceTier)
	}

	// Create completion
	ctx := c.Request.Context()
	completion, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Convert to OpenAI-compatible format using marshal/unmarshal to preserve all fields
	// This ensures we don't drop tool_calls, refusal, or any future fields
	completionJSON, err := json.Marshal(completion)
	if err != nil {
		log.Printf("Failed to marshal completion: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process completion"})
		return
	}

	var response ChatCompletionResponse
	if err := json.Unmarshal(completionJSON, &response); err != nil {
		log.Printf("Failed to unmarshal completion: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process completion"})
		return
	}

	// Override model with the requested model name
	response.Model = req.Model
	response.Object = "chat.completion"

	c.JSON(http.StatusOK, response)
}

func (s *TinfoilProxyServer) handleTranscription(c *gin.Context) {
	// Parse multipart form data
	file, fileHeader, err := c.Request.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Audio file is required"})
		return
	}
	defer file.Close()

	log.Printf("Transcription request - file: %s, size: %d bytes", fileHeader.Filename, fileHeader.Size)

	// Get other form parameters
	model := c.PostForm("model")
	if model == "" {
		model = "whisper-large-v3-turbo" // Default to Whisper Large V3 Turbo
	}

	language := c.PostForm("language")
	prompt := c.PostForm("prompt")
	responseFormat := c.PostForm("response_format")
	if responseFormat == "" {
		responseFormat = "json"
	}
	temperature := c.PostForm("temperature")

	client, err := s.getClient()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Read file content
	fileBytes, err := io.ReadAll(file)
	if err != nil {
		log.Printf("Failed to read audio file: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read audio file"})
		return
	}

	// Build transcription params
	params := openai.AudioTranscriptionNewParams{
		Model: "whisper-large-v3-turbo", // Always use Whisper Large V3 Turbo for Tinfoil
		File:  openai.File(bytes.NewReader(fileBytes), fileHeader.Filename, fileHeader.Header.Get("Content-Type")),
	}

	// Add optional parameters
	if language != "" {
		params.Language = openai.String(language)
	}
	if prompt != "" {
		params.Prompt = openai.String(prompt)
	}
	// Note: ResponseFormat handling might need adjustment based on SDK version
	// For now, we'll skip the response format parameter as it may not be available
	// in the current SDK version. The default JSON format should work.

	if temperature != "" {
		if temp, err := strconv.ParseFloat(temperature, 64); err == nil {
			params.Temperature = openai.Float(temp)
		}
	}

	// Create transcription
	ctx := c.Request.Context()
	transcription, err := client.Audio.Transcriptions.New(ctx, params)
	if err != nil {
		log.Printf("Transcription error: %v", err)

		// Check if this is a certificate error and reinitialize if needed
		if isCertificateError(err) {
			go func() {
				if reinitErr := s.reinitializeClient(); reinitErr != nil {
					log.Printf("Failed to reinitialize client: %v", reinitErr)
				}
			}()
		}

		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// TODO: Add SQS-based billing events for transcription usage tracking
	// Should track: audio duration, model used, user ID, timestamp

	// Return OpenAI-compatible response
	// The transcription object already has the text field
	response := TranscriptionResponse{
		Text: transcription.Text,
	}

	c.JSON(http.StatusOK, response)
}

func (s *TinfoilProxyServer) handleTTS(c *gin.Context) {
	var req TTSRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	log.Printf("TTS request for model: %s, voice: %s", req.Model, req.Voice)

	client, err := s.getClient()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Use voice directly without mapping
	voice := req.Voice
	if voice == "" {
		voice = "af_sky" // Default Kokoro voice
	}

	// Default values
	if req.ResponseFormat == "" {
		req.ResponseFormat = "mp3"
	}
	if req.Speed == 0 {
		req.Speed = 1.0
	}

	// Map response format to OpenAI SDK format enum
	var responseFormat openai.AudioSpeechNewParamsResponseFormat
	switch req.ResponseFormat {
	case "mp3":
		responseFormat = openai.AudioSpeechNewParamsResponseFormatMP3
	case "opus":
		responseFormat = openai.AudioSpeechNewParamsResponseFormatOpus
	case "aac":
		responseFormat = openai.AudioSpeechNewParamsResponseFormatAAC
	case "flac":
		responseFormat = openai.AudioSpeechNewParamsResponseFormatFLAC
	case "wav":
		responseFormat = openai.AudioSpeechNewParamsResponseFormatWAV
	case "pcm":
		// PCM is not supported by OpenAI's TTS API
		c.JSON(http.StatusBadRequest, gin.H{"error": "PCM format is not supported. Please use mp3, wav, opus, aac, or flac."})
		return
	default:
		responseFormat = openai.AudioSpeechNewParamsResponseFormatMP3
	}

	// Build TTS params using Kokoro model
	params := openai.AudioSpeechNewParams{
		Model:          "kokoro",                                // Always use Kokoro for TTS
		Voice:          openai.AudioSpeechNewParamsVoice(voice), // Cast string to voice type
		Input:          req.Input,
		ResponseFormat: responseFormat,
		Speed:          openai.Float(float64(req.Speed)),
	}

	// Create speech
	ctx := c.Request.Context()
	response, err := client.Audio.Speech.New(ctx, params)
	if err != nil {
		log.Printf("TTS error: %v", err)

		// Check if this is a certificate error and reinitialize if needed
		if isCertificateError(err) {
			go func() {
				if reinitErr := s.reinitializeClient(); reinitErr != nil {
					log.Printf("Failed to reinitialize client: %v", reinitErr)
				}
			}()
		}

		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer response.Body.Close()

	// Set appropriate content type based on format
	contentType := "audio/mpeg" // default to mp3
	switch req.ResponseFormat {
	case "opus":
		contentType = "audio/opus"
	case "aac":
		contentType = "audio/aac"
	case "flac":
		contentType = "audio/flac"
	case "wav":
		contentType = "audio/wav"
	}

	// Stream the audio response back to client
	c.Header("Content-Type", contentType)
	c.Status(http.StatusOK)

	// Copy the response body to the client
	buffer := make([]byte, 4096)
	for {
		n, err := response.Body.Read(buffer)
		if n > 0 {
			if _, writeErr := c.Writer.Write(buffer[:n]); writeErr != nil {
				log.Printf("Error writing TTS response: %v", writeErr)
				return
			}
			c.Writer.Flush()
		}
		if err != nil {
			if err != io.EOF {
				log.Printf("Error reading TTS response: %v", err)
			}
			break
		}
	}
}

func main() {
	// Initialize proxy server
	server, err := NewTinfoilProxyServer()
	if err != nil {
		log.Fatalf("Failed to initialize proxy server: %v", err)
	}

	// Set up Gin router
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// Health check endpoint
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "healthy",
			"service": "tinfoil-proxy",
		})
	})

	// List models endpoint
	r.GET("/v1/models", func(c *gin.Context) {
		models := []ModelInfo{}
		for modelID, config := range modelConfigs {
			if config.Active {
				models = append(models, ModelInfo{
					ID:      modelID,
					Object:  "model",
					Created: 1700000000,
					OwnedBy: "tinfoil",
				})
			}
		}
		c.JSON(http.StatusOK, ModelsResponse{
			Object: "list",
			Data:   models,
		})
	})

	// Chat completions endpoint
	r.POST("/v1/chat/completions", func(c *gin.Context) {
		var req ChatCompletionRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		log.Printf("Chat completion request for model: %s, streaming: %v", req.Model, req.Stream != nil && *req.Stream)

		// Default to non-streaming if not specified
		isStreaming := req.Stream != nil && *req.Stream

		if isStreaming {
			server.streamChatCompletion(c, req)
		} else {
			server.nonStreamingChatCompletion(c, req)
		}
	})

	// TTS endpoint
	r.POST("/v1/audio/speech", func(c *gin.Context) {
		server.handleTTS(c)
	})

	// Transcription endpoint
	r.POST("/v1/audio/transcriptions", func(c *gin.Context) {
		server.handleTranscription(c)
	})

	// Start server
	port := os.Getenv("TINFOIL_PROXY_PORT")
	if port == "" {
		port = "8093"
	}

	log.Printf("Tinfoil proxy server starting on port %s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
