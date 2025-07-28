package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/tinfoilsh/tinfoil-go"
)

// Model configurations
var modelConfigs = map[string]struct {
	ModelID     string
	Description string
	Active      bool
}{
	"deepseek-r1-70b": {
		ModelID:     "deepseek-r1-70b",
		Description: "Advanced reasoning and complex problem-solving model",
		Active:      true,
	},
	"mistral-small-3-1-24b": {
		ModelID:     "mistral-small-3-1-24b",
		Description: "Vision capabilities for image analysis, efficient performance",
		Active:      true,
	},
	"llama3-3-70b": {
		ModelID:     "llama3-3-70b",
		Description: "Multilingual understanding, dialogue optimization",
		Active:      true,
	},
	"qwen2-5-72b": {
		ModelID:     "qwen2-5-72b",
		Description: "Exceptional function calling, multilingual capabilities",
		Active:      true,
	},
	"nomic-embed-text": {
		ModelID:     "nomic-embed-text",
		Description: "Text embedding model",
		Active:      false,
	},
}

// Request/Response models
type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type ChatCompletionRequest struct {
	Model             string           `json:"model"`
	Messages          []ChatMessage    `json:"messages"`
	Stream            *bool            `json:"stream,omitempty"`
	Temperature       *float32         `json:"temperature,omitempty"`
	MaxTokens         *int             `json:"max_tokens,omitempty"`
	TopP              *float32         `json:"top_p,omitempty"`
	FrequencyPenalty  *float32         `json:"frequency_penalty,omitempty"`
	PresencePenalty   *float32         `json:"presence_penalty,omitempty"`
	N                 *int             `json:"n,omitempty"`
	Stop              []string         `json:"stop,omitempty"`
	StreamOptions     *map[string]any  `json:"stream_options,omitempty"`
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
	Index        int         `json:"index"`
	Message      *ChatMessage `json:"message,omitempty"`
	Delta        *ChatMessage `json:"delta,omitempty"`
	FinishReason *string     `json:"finish_reason"`
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

type TinfoilProxyServer struct {
	client        *tinfoil.Client
	clientMutex   sync.RWMutex
	apiKey        string
	lastRotation  time.Time
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
				Index: 0,
				Delta: &ChatMessage{},
				FinishReason: &finishReason,
			}
			chunkData.Choices = append(chunkData.Choices, choiceData)
			hasFinishReason = true
		} else {
			for _, choice := range chunk.Choices {
				choiceData := Choice{
					Index: int(choice.Index),
					Delta: &ChatMessage{},
				}
				
				if choice.Delta.Role != "" {
					choiceData.Delta.Role = string(choice.Delta.Role)
				}
				if choice.Delta.Content != "" {
					choiceData.Delta.Content = choice.Delta.Content
				} else if choice.Delta.Content == "" && choice.FinishReason == "" {
					// Tinfoil sends empty content with no finish reason - interpret as end
					log.Printf("Empty content with no finish_reason - setting finish_reason to 'stop'")
					finishReason := "stop"
					choiceData.FinishReason = &finishReason
					hasFinishReason = true
				}
				
				if choice.FinishReason != "" {
					finishReason := string(choice.FinishReason)
					choiceData.FinishReason = &finishReason
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

	// Create completion
	ctx := c.Request.Context()
	completion, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Convert to OpenAI-compatible format
	response := ChatCompletionResponse{
		ID:      completion.ID,
		Object:  "chat.completion",
		Created: completion.Created,
		Model:   req.Model,
		Choices: make([]Choice, 0),
	}

	for _, choice := range completion.Choices {
		finishReason := string(choice.FinishReason)
		choiceData := Choice{
			Index: int(choice.Index),
			Message: &ChatMessage{
				Role:    string(choice.Message.Role),
				Content: choice.Message.Content,
			},
			FinishReason: &finishReason,
		}
		response.Choices = append(response.Choices, choiceData)
	}

	// Include usage data
	// Note: Usage is a struct, not a pointer in the OpenAI SDK
	if completion.Usage.TotalTokens > 0 {
		response.Usage = &Usage{
			PromptTokens:     int(completion.Usage.PromptTokens),
			CompletionTokens: int(completion.Usage.CompletionTokens),
			TotalTokens:      int(completion.Usage.TotalTokens),
		}
	}

	c.JSON(http.StatusOK, response)
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
			"status": "healthy",
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
