use crate::ApiError;
use serde_json::from_str;
use unicode_segmentation::UnicodeSegmentation;

const MAX_REACTION_CHARS: usize = 16;

fn canonicalize_optional_reaction(value: Option<&str>) -> Option<String> {
    let mut current = value.map(str::trim)?.to_string();
    if current.is_empty() {
        return None;
    }

    for _ in 0..2 {
        let trimmed = current.trim();
        if !(trimmed.starts_with('"') && trimmed.ends_with('"')) {
            break;
        }

        let Ok(decoded) = from_str::<String>(trimmed) else {
            break;
        };

        let decoded = decoded.trim();
        if decoded.is_empty() {
            return None;
        }

        current = decoded.to_string();
    }

    Some(current.trim().to_string())
}

pub(super) fn has_meaningful_reaction_candidate(value: &str) -> bool {
    canonicalize_optional_reaction(Some(value)).is_some()
}

pub(super) fn normalize_optional_reaction(value: Option<&str>) -> Option<String> {
    let trimmed = canonicalize_optional_reaction(value)?;

    if trimmed.chars().count() > MAX_REACTION_CHARS {
        return None;
    }

    if trimmed.graphemes(true).count() != 1 {
        return None;
    }

    if !trimmed.chars().all(is_allowed_reaction_char) {
        return None;
    }

    if !has_visible_emoji_base(&trimmed) {
        return None;
    }

    Some(trimmed.to_string())
}

pub(super) fn require_valid_reaction(value: &str) -> Result<String, ApiError> {
    normalize_optional_reaction(Some(value)).ok_or(ApiError::BadRequest)
}

fn is_allowed_reaction_char(ch: char) -> bool {
    matches!(
        ch,
        '#' | '*' | '0'..='9'
            | '\u{00A9}'
            | '\u{00AE}'
            | '\u{200D}'
            | '\u{203C}'
            | '\u{2049}'
            | '\u{20E3}'
            | '\u{2122}'
            | '\u{2139}'
            | '\u{2194}'..='\u{2199}'
            | '\u{21A9}'..='\u{21AA}'
            | '\u{231A}'..='\u{231B}'
            | '\u{2328}'
            | '\u{23CF}'
            | '\u{23E9}'..='\u{23F3}'
            | '\u{23F8}'..='\u{23FA}'
            | '\u{24C2}'
            | '\u{25AA}'..='\u{25AB}'
            | '\u{25B6}'
            | '\u{25C0}'
            | '\u{25FB}'..='\u{25FE}'
            | '\u{2600}'..='\u{27BF}'
            | '\u{2934}'..='\u{2935}'
            | '\u{2B05}'..='\u{2B07}'
            | '\u{2B1B}'..='\u{2B1C}'
            | '\u{2B50}'
            | '\u{2B55}'
            | '\u{3030}'
            | '\u{303D}'
            | '\u{3297}'
            | '\u{3299}'
            | '\u{FE0E}'
            | '\u{FE0F}'
            | '\u{1F000}'..='\u{1FAFF}'
            | '\u{E0020}'..='\u{E007F}'
    )
}

fn has_visible_emoji_base(value: &str) -> bool {
    has_keycap_base(value) || value.chars().any(is_non_component_emoji_char)
}

fn has_keycap_base(value: &str) -> bool {
    value.chars().any(|ch| matches!(ch, '#' | '*' | '0'..='9'))
        && value.chars().any(|ch| ch == '\u{20E3}')
}

fn is_non_component_emoji_char(ch: char) -> bool {
    is_allowed_reaction_char(ch) && !is_component_only_reaction_char(ch)
}

fn is_component_only_reaction_char(ch: char) -> bool {
    matches!(
        ch,
        '#' | '*' | '0'..='9'
            | '\u{200D}'
            | '\u{20E3}'
            | '\u{FE0E}'
            | '\u{FE0F}'
            | '\u{1F3FB}'..='\u{1F3FF}'
            | '\u{E0020}'..='\u{E007F}'
    )
}

#[cfg(test)]
mod tests {
    use super::{
        has_meaningful_reaction_candidate, normalize_optional_reaction, require_valid_reaction,
    };

    #[test]
    fn accepts_common_emoji_reactions() {
        assert_eq!(
            normalize_optional_reaction(Some("❤️")),
            Some("❤️".to_string())
        );
        assert_eq!(
            normalize_optional_reaction(Some("🫡")),
            Some("🫡".to_string())
        );
        assert_eq!(
            normalize_optional_reaction(Some("1️⃣")),
            Some("1️⃣".to_string())
        );
        assert_eq!(
            normalize_optional_reaction(Some("🇺🇸")),
            Some("🇺🇸".to_string())
        );
    }

    #[test]
    fn trims_whitespace() {
        assert_eq!(
            normalize_optional_reaction(Some("  🎉  ")),
            Some("🎉".to_string())
        );
    }

    #[test]
    fn accepts_complex_modifier_and_zwj_emoji_reactions() {
        assert_eq!(
            normalize_optional_reaction(Some("🤦🏽‍♂️")),
            Some("🤦🏽‍♂️".to_string())
        );
        assert_eq!(
            normalize_optional_reaction(Some("🤷🏼‍♀️")),
            Some("🤷🏼‍♀️".to_string())
        );
        assert_eq!(
            normalize_optional_reaction(Some("🧑🏾‍💻")),
            Some("🧑🏾‍💻".to_string())
        );
    }

    #[test]
    fn rejects_non_emoji_text() {
        assert_eq!(normalize_optional_reaction(Some("hello")), None);
        assert!(require_valid_reaction("not-an-emoji").is_err());
    }

    #[test]
    fn rejects_component_only_values() {
        assert_eq!(normalize_optional_reaction(Some("1")), None);
        assert_eq!(normalize_optional_reaction(Some("#")), None);
        assert_eq!(normalize_optional_reaction(Some("*")), None);
        assert_eq!(normalize_optional_reaction(Some("\u{200D}")), None);
        assert_eq!(normalize_optional_reaction(Some("\u{20E3}")), None);
        assert_eq!(normalize_optional_reaction(Some("🏽")), None);
    }

    #[test]
    fn rejects_injection_style_values() {
        assert_eq!(normalize_optional_reaction(Some("]: hacked")), None);
        assert_eq!(normalize_optional_reaction(Some("❤️ hi")), None);
        assert_eq!(normalize_optional_reaction(Some("❤️🎉")), None);
    }

    #[test]
    fn unwraps_stringified_reaction_values() {
        assert_eq!(
            normalize_optional_reaction(Some("\"❤️\"")),
            Some("❤️".to_string())
        );
        assert_eq!(normalize_optional_reaction(Some("\"\"")), None);
    }

    #[test]
    fn detects_meaningful_reaction_candidates_after_unquoting() {
        assert!(has_meaningful_reaction_candidate("\"🎉\""));
        assert!(!has_meaningful_reaction_candidate("\"\""));
    }
}
