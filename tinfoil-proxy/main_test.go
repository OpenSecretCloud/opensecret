package main

import (
	"encoding/json"
	"testing"
)

func TestBuildExtraChatCompletionFieldsForKimiThinkingControl(t *testing.T) {
	includeReasoning := false
	thinkingTokenBudget := 0

	req := ChatCompletionRequest{
		Thinking:            map[string]any{"type": "disabled"},
		IncludeReasoning:    &includeReasoning,
		ThinkingTokenBudget: &thinkingTokenBudget,
		ChatTemplateKwargs:  map[string]any{"thinking": false},
	}

	extraFields := buildExtraChatCompletionFields(req)
	if len(extraFields) != 4 {
		t.Fatalf("expected 4 extra fields, got %d", len(extraFields))
	}

	thinking, ok := extraFields["thinking"].(map[string]any)
	if !ok || thinking["type"] != "disabled" {
		t.Fatalf("unexpected thinking field: %#v", extraFields["thinking"])
	}

	includeReasoningValue, ok := extraFields["include_reasoning"].(bool)
	if !ok || includeReasoningValue {
		t.Fatalf("unexpected include_reasoning field: %#v", extraFields["include_reasoning"])
	}

	thinkingTokenBudgetValue, ok := extraFields["thinking_token_budget"].(int)
	if !ok || thinkingTokenBudgetValue != 0 {
		t.Fatalf("unexpected thinking_token_budget field: %#v", extraFields["thinking_token_budget"])
	}

	chatTemplateKwargs, ok := extraFields["chat_template_kwargs"].(map[string]any)
	if !ok || chatTemplateKwargs["thinking"] != false {
		t.Fatalf("unexpected chat_template_kwargs field: %#v", extraFields["chat_template_kwargs"])
	}
}

func TestBuildExtraChatCompletionFieldsForGemmaThinkingMode(t *testing.T) {
	includeReasoning := true

	req := ChatCompletionRequest{
		IncludeReasoning:   &includeReasoning,
		ChatTemplateKwargs: map[string]any{"enable_thinking": true},
	}

	extraFields := buildExtraChatCompletionFields(req)
	if len(extraFields) != 2 {
		t.Fatalf("expected 2 extra fields, got %d", len(extraFields))
	}

	includeReasoningValue, ok := extraFields["include_reasoning"].(bool)
	if !ok || !includeReasoningValue {
		t.Fatalf("unexpected include_reasoning field: %#v", extraFields["include_reasoning"])
	}

	chatTemplateKwargs, ok := extraFields["chat_template_kwargs"].(map[string]any)
	if !ok || chatTemplateKwargs["enable_thinking"] != true {
		t.Fatalf("unexpected chat_template_kwargs field: %#v", extraFields["chat_template_kwargs"])
	}
}

func TestBuildExtraChatCompletionFieldsEmpty(t *testing.T) {
	extraFields := buildExtraChatCompletionFields(ChatCompletionRequest{})
	if extraFields != nil {
		t.Fatalf("expected nil extra fields, got %#v", extraFields)
	}
}

func TestMarshalChatCompletionRequestIncludesStandardReasoningControls(t *testing.T) {
	// GPT-OSS uses the standard OpenAI-style request fields for reasoning control.
	// On the current Tinfoil/vLLM stack, callers can tune reasoning with
	// `reasoning_effort` and `verbosity`, while Kimi's vLLM-specific no-thinking
	// mode uses `chat_template_kwargs.thinking=false`.
	reasoningEffort := "low"
	verbosity := "low"

	req := ChatCompletionRequest{
		Model:              "gpt-oss-120b",
		Messages:           []ChatMessage{{Role: "user", Content: "hi"}},
		ReasoningEffort:    &reasoningEffort,
		Verbosity:          &verbosity,
		ChatTemplateKwargs: nil,
	}

	if extraFields := buildExtraChatCompletionFields(req); extraFields != nil {
		t.Fatalf("expected no extra fields for standard reasoning controls, got %#v", extraFields)
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}

	if payload["reasoning_effort"] != "low" {
		t.Fatalf("expected reasoning_effort=low, got %#v", payload["reasoning_effort"])
	}

	if payload["verbosity"] != "low" {
		t.Fatalf("expected verbosity=low, got %#v", payload["verbosity"])
	}

	if _, ok := payload["chat_template_kwargs"]; ok {
		t.Fatalf("did not expect chat_template_kwargs in payload: %#v", payload["chat_template_kwargs"])
	}
}

func TestMarshalChatCompletionRequestIncludesGemmaThinkingControls(t *testing.T) {
	includeReasoning := true

	req := ChatCompletionRequest{
		Model:              "gemma4-31b",
		Messages:           []ChatMessage{{Role: "user", Content: "hi"}},
		IncludeReasoning:   &includeReasoning,
		ChatTemplateKwargs: map[string]any{"enable_thinking": true},
	}

	if extraFields := buildExtraChatCompletionFields(req); len(extraFields) != 2 {
		t.Fatalf("expected gemma extra fields, got %#v", extraFields)
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("failed to marshal request: %v", err)
	}

	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		t.Fatalf("failed to unmarshal request payload: %v", err)
	}

	if payload["include_reasoning"] != true {
		t.Fatalf("expected include_reasoning=true, got %#v", payload["include_reasoning"])
	}

	chatTemplateKwargs, ok := payload["chat_template_kwargs"].(map[string]any)
	if !ok || chatTemplateKwargs["enable_thinking"] != true {
		t.Fatalf("unexpected chat_template_kwargs in payload: %#v", payload["chat_template_kwargs"])
	}
}

func TestModelConfigsIncludesGemma4_31B(t *testing.T) {
	config, ok := modelConfigs["gemma4-31b"]
	if !ok {
		t.Fatal("expected gemma4-31b to be registered")
	}

	if config.ModelID != "gemma4-31b" {
		t.Fatalf("expected model id gemma4-31b, got %q", config.ModelID)
	}

	if !config.Active {
		t.Fatal("expected gemma4-31b to be active")
	}
}
