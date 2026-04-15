package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3/option"
	"github.com/tinfoilsh/tinfoil-go"
)

var errMissingAPIKey = errors.New("TINFOIL_API_KEY environment variable is required")

const upstreamResponseStartTimeout = 120 * time.Second
const defaultWebSearchAPIBase = "https://websearch-debug.debug.tinfoil.containers.tinfoil.dev"

type proxyServer struct {
	httpClient      *http.Client
	webSearchClient *http.Client
	apiKey          string
	enclaveHost     string
	webSearchBase   *url.URL
}

type upstreamTarget struct {
	client  *http.Client
	baseURL url.URL
	name    string
}

type chatCompletionEnvelope struct {
	WebSearchOptions *json.RawMessage `json:"web_search_options"`
}

type flushWriter struct {
	writer  io.Writer
	flusher http.Flusher
}

func (w flushWriter) Write(p []byte) (int, error) {
	n, err := w.writer.Write(p)
	if n > 0 {
		w.flusher.Flush()
	}
	return n, err
}

func newProxyServer() (*proxyServer, error) {
	apiKey := os.Getenv("TINFOIL_API_KEY")
	if apiKey == "" {
		return nil, errMissingAPIKey
	}

	client, err := tinfoil.NewClient(option.WithAPIKey(apiKey))
	if err != nil {
		return nil, err
	}

	httpClient := *client.HTTPClient()
	httpClient.Timeout = 0

	webSearchBase := os.Getenv("TINFOIL_WEBSEARCH_API_BASE")
	if webSearchBase == "" {
		webSearchBase = defaultWebSearchAPIBase
	}

	parsedWebSearchBase, err := url.Parse(webSearchBase)
	if err != nil {
		return nil, err
	}
	if parsedWebSearchBase.Scheme == "" || parsedWebSearchBase.Host == "" {
		return nil, errors.New("TINFOIL_WEBSEARCH_API_BASE must be an absolute URL")
	}

	return &proxyServer{
		httpClient:      &httpClient,
		webSearchClient: &http.Client{},
		apiKey:          apiKey,
		enclaveHost:     client.Enclave(),
		webSearchBase:   parsedWebSearchBase,
	}, nil
}

func (s *proxyServer) upstreamURL(path, rawQuery string) string {
	return buildUpstreamURL(url.URL{
		Scheme: "https",
		Host:   s.enclaveHost,
	}, path, rawQuery)
}

func buildUpstreamURL(base url.URL, path, rawQuery string) string {
	upstream := base
	upstream.Path = joinURLPath(base.Path, path)
	upstream.RawQuery = rawQuery
	return upstream.String()
}

func joinURLPath(basePath, path string) string {
	trimmedPath := strings.TrimLeft(path, "/")
	if basePath == "" || basePath == "/" {
		return "/" + trimmedPath
	}

	return strings.TrimRight(basePath, "/") + "/" + trimmedPath
}

func shouldUseWebSearchUpstream(body []byte) bool {
	if len(bytes.TrimSpace(body)) == 0 {
		return false
	}

	var envelope chatCompletionEnvelope
	if err := json.Unmarshal(body, &envelope); err != nil {
		return false
	}
	if envelope.WebSearchOptions == nil {
		return false
	}

	return !bytes.Equal(bytes.TrimSpace(*envelope.WebSearchOptions), []byte("null"))
}

func (s *proxyServer) resolveUpstreamTarget(path string, body []byte) upstreamTarget {
	if path == "/v1/chat/completions" && shouldUseWebSearchUpstream(body) {
		return upstreamTarget{
			client:  s.webSearchClient,
			baseURL: *s.webSearchBase,
			name:    "websearch",
		}
	}

	return upstreamTarget{
		client: s.httpClient,
		baseURL: url.URL{
			Scheme: "https",
			Host:   s.enclaveHost,
		},
		name: "enclave",
	}
}

func shouldSkipHeader(name string) bool {
	switch http.CanonicalHeaderKey(name) {
	case "Authorization", "Connection", "Content-Length", "Host", "Keep-Alive",
		"Proxy-Authenticate", "Proxy-Authorization", "Te", "Trailer",
		"Transfer-Encoding", "Upgrade":
		return true
	default:
		return false
	}
}

func headerSkipSet(src http.Header) map[string]struct{} {
	skip := make(map[string]struct{})

	for key := range src {
		if shouldSkipHeader(key) {
			skip[http.CanonicalHeaderKey(key)] = struct{}{}
		}
	}

	for _, value := range src.Values("Connection") {
		for _, token := range strings.Split(value, ",") {
			token = strings.TrimSpace(token)
			if token == "" {
				continue
			}
			skip[http.CanonicalHeaderKey(token)] = struct{}{}
		}
	}

	return skip
}

func copyHeaders(dst, src http.Header) {
	skip := headerSkipSet(src)

	for key, values := range src {
		if _, ok := skip[http.CanonicalHeaderKey(key)]; ok {
			continue
		}
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func writeProxyError(c *gin.Context, status int, message string) {
	c.AbortWithStatusJSON(status, gin.H{
		"error": gin.H{
			"message": message,
			"type":    "server_error",
		},
	})
}

type proxyResult struct {
	resp *http.Response
	err  error
}

func doWithResponseStartTimeout(
	client *http.Client,
	req *http.Request,
	timeout time.Duration,
) (*http.Response, error) {
	if timeout <= 0 {
		return client.Do(req)
	}

	ctx, cancel := context.WithCancel(req.Context())
	req = req.Clone(ctx)

	resultCh := make(chan proxyResult, 1)
	go func() {
		resp, err := client.Do(req)
		resultCh <- proxyResult{resp: resp, err: err}
	}()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case result := <-resultCh:
		return result.resp, result.err
	case <-timer.C:
		select {
		case result := <-resultCh:
			return result.resp, result.err
		default:
		}

		cancel()
		result := <-resultCh
		if result.resp != nil {
			result.resp.Body.Close()
		}
		return nil, context.DeadlineExceeded
	}
}

func (s *proxyServer) proxy(c *gin.Context, path string) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		log.Printf("failed to read request body for %s: %v", path, err)
		writeProxyError(c, http.StatusBadRequest, "failed to read request body")
		return
	}

	target := s.resolveUpstreamTarget(path, body)
	upstreamURL := buildUpstreamURL(target.baseURL, path, c.Request.URL.RawQuery)

	if target.name == "websearch" {
		log.Printf("routing %s to experimental web search upstream %s", path, target.baseURL.Host)
	}

	req, err := http.NewRequestWithContext(
		c.Request.Context(),
		c.Request.Method,
		upstreamURL,
		bytes.NewReader(body),
	)
	if err != nil {
		log.Printf("failed to create upstream request for %s: %v", path, err)
		writeProxyError(c, http.StatusInternalServerError, "failed to create upstream request")
		return
	}

	copyHeaders(req.Header, c.Request.Header)
	req.Header.Set("Authorization", "Bearer "+s.apiKey)
	req.Host = target.baseURL.Host

	resp, err := doWithResponseStartTimeout(target.client, req, upstreamResponseStartTimeout)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			log.Printf("upstream response start timed out for %s after %s", path, upstreamResponseStartTimeout)
		} else {
			log.Printf("upstream request failed for %s: %v", path, err)
		}
		writeProxyError(c, http.StatusBadGateway, "failed to reach tinfoil upstream")
		return
	}
	defer resp.Body.Close()

	copyHeaders(c.Writer.Header(), resp.Header)
	c.Status(resp.StatusCode)

	if flusher, ok := c.Writer.(http.Flusher); ok {
		if _, err := io.Copy(flushWriter{writer: c.Writer, flusher: flusher}, resp.Body); err != nil {
			log.Printf("failed to stream upstream response for %s: %v", path, err)
		}
		return
	}

	if _, err := io.Copy(c.Writer, resp.Body); err != nil {
		log.Printf("failed to copy upstream response for %s: %v", path, err)
	}
}

func (s *proxyServer) proxyPath(path string) gin.HandlerFunc {
	return func(c *gin.Context) {
		s.proxy(c, path)
	}
}

func main() {
	server, err := newProxyServer()
	if err != nil {
		log.Fatalf("failed to initialize proxy server: %v", err)
	}

	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery())

	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "healthy",
			"service": "tinfoil-proxy",
		})
	})

	router.GET("/v1/models", server.proxyPath("/v1/models"))
	router.POST("/v1/chat/completions", server.proxyPath("/v1/chat/completions"))
	router.POST("/v1/audio/speech", server.proxyPath("/v1/audio/speech"))
	router.POST("/v1/audio/transcriptions", server.proxyPath("/v1/audio/transcriptions"))
	router.POST("/v1/embeddings", server.proxyPath("/v1/embeddings"))

	port := os.Getenv("TINFOIL_PROXY_PORT")
	if port == "" {
		port = "8093"
	}

	log.Printf("tinfoil proxy listening on %s", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("failed to start proxy server: %v", err)
	}
}
