package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
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

// Document upload service configuration
var docUploadConfig = struct {
	Enclave string
	Repo    string
}{
	Enclave: "doc-upload.model.tinfoil.sh",
	Repo:    "tinfoilsh/confidential-doc-upload",
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

type DocumentUploadRequest struct {
	Filename      string `json:"filename"`
	ContentBase64 string `json:"content_base64"`
}

type DocumentUploadResponse struct {
	Text     string `json:"text"`
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
}

type AsyncJobResponse struct {
	TaskID string `json:"task_id"`
}

type JobStatus struct {
	Status   string  `json:"status"` // pending, started, success, failure
	Progress *int    `json:"progress,omitempty"`
	Error    *string `json:"error,omitempty"`
	// Also try common variations
	State    string  `json:"state,omitempty"`
	TaskStatus string `json:"task_status,omitempty"`
}

type DocumentStatusResponse struct {
	Status   string                  `json:"status"`
	Progress *int                    `json:"progress,omitempty"`
	Error    *string                 `json:"error,omitempty"`
	Document *DocumentUploadResponse `json:"document,omitempty"`
}

type TinfoilProxyServer struct {
	clients       map[string]*tinfoil.Client
	docUploadClient *tinfoil.Client
	docUploadSecureClient *tinfoil.SecureClient
}

func NewTinfoilProxyServer() (*TinfoilProxyServer, error) {
	apiKey := os.Getenv("TINFOIL_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("TINFOIL_API_KEY environment variable is required")
	}

	server := &TinfoilProxyServer{
		clients: make(map[string]*tinfoil.Client),
	}

	// Initialize Tinfoil client with new simplified API
	log.Printf("Initializing Tinfoil client with new API")
	
	// Create a single client that will handle all models through the inference endpoint
	client, err := tinfoil.NewClient(option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Tinfoil client: %v", err)
	}
	
	// Register all active models
	for modelName, config := range modelConfigs {
		if !config.Active {
			log.Printf("Skipping inactive model: %s", modelName)
			continue
		}

		log.Printf("Registering model %s", modelName)
		
		// Use the same client for all models - the inference endpoint will route based on model name
		server.clients[modelName] = client
		log.Printf("Successfully registered model: %s", modelName)
	}

	// Initialize document upload service
	log.Printf("Initializing document upload service")
	// Document upload still uses the specific enclave URL
	docClient, err := tinfoil.NewClientWithParams(
		docUploadConfig.Enclave,
		docUploadConfig.Repo,
		option.WithAPIKey(apiKey),
	)
	if err != nil {
		log.Printf("Failed to initialize document upload service: %v", err)
		// Don't fail if doc upload service can't be initialized
	} else {
		server.docUploadClient = docClient
		// Also create a SecureClient for HTTP requests
		// Note: NewSecureClient now uses simplified API without parameters
		server.docUploadSecureClient = tinfoil.NewSecureClient()
		
		// Verify the enclave
		_, err = server.docUploadSecureClient.Verify()
		if err != nil {
			log.Printf("Failed to verify document upload enclave: %v", err)
			server.docUploadSecureClient = nil
		} else {
			log.Printf("Successfully verified document upload enclave")
		}
		
		log.Printf("Successfully initialized document upload service")
	}

	if len(server.clients) == 0 {
		return nil, fmt.Errorf("no models could be initialized")
	}

	return server, nil
}

func (s *TinfoilProxyServer) getClient(model string) (*tinfoil.Client, error) {
	client, ok := s.clients[model]
	if !ok {
		return nil, fmt.Errorf("model '%s' not supported", model)
	}
	return client, nil
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
	client, err := s.getClient(req.Model)
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

	// Set up SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	// Create context for cancellation
	ctx := c.Request.Context()

	// Start streaming
	stream := client.Chat.Completions.NewStreaming(ctx, params)
	defer stream.Close()
	
	// Stream responses
	w := c.Writer
	flusher, ok := w.(http.Flusher)
	if !ok {
		log.Printf("Response writer does not support flushing")
		return
	}

	for stream.Next() {
		chunk := stream.Current()
		
		// Convert to OpenAI-compatible format
		chunkData := ChatCompletionResponse{
			ID:      chunk.ID,
			Object:  "chat.completion.chunk",
			Created: chunk.Created,
			Model:   req.Model,
			Choices: make([]Choice, 0),
		}

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
				}
				
				if choice.FinishReason != "" {
					finishReason := string(choice.FinishReason)
					choiceData.FinishReason = &finishReason
				}

				chunkData.Choices = append(chunkData.Choices, choiceData)
			}
		}

		// Include usage data if available
		// Note: Usage is a struct, not a pointer in the OpenAI SDK
		if chunk.Usage.TotalTokens > 0 {
			chunkData.Usage = &Usage{
				PromptTokens:     int(chunk.Usage.PromptTokens),
				CompletionTokens: int(chunk.Usage.CompletionTokens),
				TotalTokens:      int(chunk.Usage.TotalTokens),
			}
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
		// Don't send error details to client since they can't handle it properly
	}

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *TinfoilProxyServer) uploadDocument(c *gin.Context) {
	// Get the uploaded file
	file, header, err := c.Request.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}
	defer file.Close()

	// Check file size (limit to 10MB)
	if header.Size > 10*1024*1024 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "File size exceeds 10MB limit"})
		return
	}

	// Verify that document upload service is available
	if s.docUploadSecureClient == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Document upload service not available"})
		return
	}

	// Create a buffer and multipart writer for the request
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	// Create a form file field
	part, err := writer.CreateFormFile("files", header.Filename)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create form file"})
		return
	}

	// Copy the uploaded file to the form
	_, err = io.Copy(part, file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to copy file"})
		return
	}

	// Close the multipart writer
	err = writer.Close()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to close writer"})
		return
	}

	// Create the request to Tinfoil document upload service - now using async endpoint
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	req, err := http.NewRequestWithContext(ctx, "POST", "https://doc-upload.model.tinfoil.sh/v1alpha/convert/file/async", &requestBody)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create request"})
		return
	}

	// Set the content type with boundary
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Get the secure HTTP client from the SecureClient
	httpClient, err := s.docUploadSecureClient.HTTPClient()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get secure HTTP client"})
		return
	}
	
	// Add API key if available
	apiKey := os.Getenv("TINFOIL_API_KEY")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	
	// Send the async request
	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("HTTP request failed: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to upload document"})
		return
	}
	defer resp.Body.Close()

	// Read the response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read response"})
		return
	}

	// Check for non-200 status
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		log.Printf("Async submission failed with status %d: %s", resp.StatusCode, string(body))
		c.JSON(resp.StatusCode, gin.H{"error": "Document processing failed"})
		return
	}

	// Parse the async job response
	var asyncResp AsyncJobResponse
	if err := json.Unmarshal(body, &asyncResp); err != nil {
		log.Printf("Failed to parse async response: %v, body: %s", err, string(body))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse async response"})
		return
	}

	if asyncResp.TaskID == "" {
		log.Printf("No task ID in async response: %s", string(body))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "No task ID in response"})
		return
	}

	log.Printf("Started async document processing with task ID: %s", asyncResp.TaskID)

	// Return the task ID immediately so frontend can poll
	c.JSON(http.StatusAccepted, gin.H{
		"task_id": asyncResp.TaskID,
		"filename": header.Filename,
		"size": header.Size,
	})
}

func (s *TinfoilProxyServer) checkDocumentStatus(c *gin.Context) {
	taskID := c.Param("taskId")
	if taskID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Task ID is required"})
		return
	}

	// Verify that document upload service is available
	if s.docUploadSecureClient == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Document upload service not available"})
		return
	}

	// Get the secure HTTP client
	httpClient, err := s.docUploadSecureClient.HTTPClient()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get secure HTTP client"})
		return
	}

	// Add API key if available
	apiKey := os.Getenv("TINFOIL_API_KEY")

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Check status
	statusReq, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("https://doc-upload.model.tinfoil.sh/v1alpha/status/poll/%s", taskID),
		nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create status request"})
		return
	}

	if apiKey != "" {
		statusReq.Header.Set("Authorization", "Bearer "+apiKey)
	}

	statusResp, err := httpClient.Do(statusReq)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to check status"})
		return
	}
	defer statusResp.Body.Close()

	statusBody, err := io.ReadAll(statusResp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read status response"})
		return
	}

	var status JobStatus
	if err := json.Unmarshal(statusBody, &status); err != nil {
		log.Printf("Failed to parse status response: %v, body: %s", err, string(statusBody))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse status response"})
		return
	}

	// Try to determine the actual status field
	actualStatus := status.Status
	if actualStatus == "" {
		actualStatus = status.State
	}
	if actualStatus == "" {
		actualStatus = status.TaskStatus
	}

	// Create response
	response := DocumentStatusResponse{
		Status:   actualStatus,
		Progress: status.Progress,
		Error:    status.Error,
	}

	// If the task is complete, fetch the result
	if actualStatus == "success" {
		resultReq, err := http.NewRequestWithContext(ctx, "GET",
			fmt.Sprintf("https://doc-upload.model.tinfoil.sh/v1alpha/result/%s", taskID),
			nil)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create result request"})
			return
		}

		if apiKey != "" {
			resultReq.Header.Set("Authorization", "Bearer "+apiKey)
		}

		resultResp, err := httpClient.Do(resultReq)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get result"})
			return
		}
		defer resultResp.Body.Close()

		resultBody, err := io.ReadAll(resultResp.Body)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read result"})
			return
		}

		// Parse the result
		var resultData map[string]interface{}
		if err := json.Unmarshal(resultBody, &resultData); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to parse result"})
			return
		}

		// Extract text from result
		text := ""
		if textValue, ok := resultData["text"]; ok {
			text = fmt.Sprintf("%v", textValue)
		} else if contentValue, ok := resultData["content"]; ok {
			text = fmt.Sprintf("%v", contentValue)
		} else {
			// Return the whole result as text
			text = string(resultBody)
		}

		// Include document in response
		response.Document = &DocumentUploadResponse{
			Text: text,
			// Note: We don't have filename and size here, frontend should store these
		}
	}

	c.JSON(http.StatusOK, response)
}

func (s *TinfoilProxyServer) nonStreamingChatCompletion(c *gin.Context, req ChatCompletionRequest) {
	client, err := s.getClient(req.Model)
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
		for modelID := range server.clients {
			models = append(models, ModelInfo{
				ID:      modelID,
				Object:  "model",
				Created: 1700000000,
				OwnedBy: "tinfoil",
			})
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

	// Document upload endpoint
	r.POST("/v1/documents/upload", server.uploadDocument)
	
	// Document status endpoint
	r.GET("/v1/documents/status/:taskId", server.checkDocumentStatus)

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
