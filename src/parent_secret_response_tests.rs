use super::{parse_parent_secret_response, Error};

#[test]
fn parses_valid_parent_secret_response() {
    let response = r#"{
        "response_type": "secret",
        "response_value": "{\"database_url\":\"postgres://localhost/test\"}"
    }"#;

    let secret = parse_parent_secret_response(response).expect("valid response should parse");

    assert_eq!(secret, "postgres://localhost/test");
}

#[test]
fn rejects_non_string_response_value_without_panicking() {
    let response = r#"{
        "response_type": "secret",
        "response_value": {"database_url": "postgres://localhost/test"}
    }"#;

    assert!(matches!(
        parse_parent_secret_response(response),
        Err(Error::SecretParsingError)
    ));
}

#[test]
fn rejects_truncated_parent_response() {
    let response = r#"{"response_type":"secret","response_value":"{}""#;

    assert!(matches!(
        parse_parent_secret_response(response),
        Err(Error::JsonError(_))
    ));
}

#[test]
fn rejects_unexpected_parent_response_type() {
    let response = r#"{"response_type":"credentials","response_value":"{}"}"#;

    assert!(matches!(
        parse_parent_secret_response(response),
        Err(Error::AuthenticationError)
    ));
}

#[test]
fn rejects_malformed_encoded_secret() {
    let response = r#"{"response_type":"secret","response_value":"not json"}"#;

    assert!(matches!(
        parse_parent_secret_response(response),
        Err(Error::JsonError(_))
    ));
}

#[test]
fn rejects_invalid_secret_object_shapes() {
    for response_value in [
        "{}",
        r#"{"database_url":42}"#,
        r#"{"database_url":"first","jwt_secret":"second"}"#,
    ] {
        let response = serde_json::json!({
            "response_type": "secret",
            "response_value": response_value,
        })
        .to_string();

        assert!(matches!(
            parse_parent_secret_response(&response),
            Err(Error::SecretParsingError)
        ));
    }
}
