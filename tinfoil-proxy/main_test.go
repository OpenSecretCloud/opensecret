package main

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"testing"
	"time"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestShouldSkipHeader(t *testing.T) {
	tests := map[string]bool{
		"Authorization":     true,
		"Connection":        true,
		"Content-Length":    true,
		"Transfer-Encoding": true,
		"Content-Type":      false,
		"X-Test":            false,
	}

	for header, want := range tests {
		if got := shouldSkipHeader(header); got != want {
			t.Fatalf("header %q: expected %v, got %v", header, want, got)
		}
	}
}

func TestCopyHeadersSkipsHopByHopAndAuthorization(t *testing.T) {
	src := http.Header{
		"Authorization":    []string{"Bearer secret"},
		"Connection":       []string{"keep-alive, X-Per-Connection", "X-Another-Hop"},
		"Content-Type":     []string{"application/json"},
		"X-Request-Id":     []string{"abc123"},
		"X-Per-Connection": []string{"do-not-forward"},
		"X-Another-Hop":    []string{"also-do-not-forward"},
	}
	dst := http.Header{}

	copyHeaders(dst, src)

	if dst.Get("Authorization") != "" {
		t.Fatal("expected authorization header to be skipped")
	}
	if dst.Get("Connection") != "" {
		t.Fatal("expected connection header to be skipped")
	}
	if dst.Get("X-Per-Connection") != "" {
		t.Fatal("expected connection-nominated header to be skipped")
	}
	if dst.Get("X-Another-Hop") != "" {
		t.Fatal("expected additional connection-nominated header to be skipped")
	}
	if dst.Get("Content-Type") != "application/json" {
		t.Fatalf("expected content-type to be copied, got %q", dst.Get("Content-Type"))
	}
	if dst.Get("X-Request-Id") != "abc123" {
		t.Fatalf("expected x-request-id to be copied, got %q", dst.Get("X-Request-Id"))
	}
}

func TestUpstreamURLPreservesQuery(t *testing.T) {
	server := &proxyServer{enclaveHost: "enclave.example.com"}

	got := server.upstreamURL("/v1/models", "limit=10&after=abc")
	want := "https://enclave.example.com/v1/models?limit=10&after=abc"

	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestShouldUseWebSearchUpstream(t *testing.T) {
	tests := []struct {
		name string
		body string
		want bool
	}{
		{
			name: "enabled",
			body: `{"model":"gemma4-31b","web_search_options":{}}`,
			want: true,
		},
		{
			name: "missing",
			body: `{"model":"gemma4-31b"}`,
			want: false,
		},
		{
			name: "null",
			body: `{"model":"gemma4-31b","web_search_options":null}`,
			want: false,
		},
		{
			name: "invalid json",
			body: `{`,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldUseWebSearchUpstream([]byte(tt.body)); got != tt.want {
				t.Fatalf("expected %v, got %v", tt.want, got)
			}
		})
	}
}

func TestBuildUpstreamURLAppendsBasePath(t *testing.T) {
	base, err := url.Parse("https://search.example.com/internal")
	if err != nil {
		t.Fatalf("failed to parse URL: %v", err)
	}

	got := buildUpstreamURL(*base, "/v1/chat/completions", "stream=false")
	want := "https://search.example.com/internal/v1/chat/completions?stream=false"

	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestResolveUpstreamTargetRoutesWebSearchRequests(t *testing.T) {
	webSearchBase, err := url.Parse("https://search.example.com")
	if err != nil {
		t.Fatalf("failed to parse URL: %v", err)
	}

	server := &proxyServer{
		httpClient:      &http.Client{},
		webSearchClient: &http.Client{},
		enclaveHost:     "enclave.example.com",
		webSearchBase:   webSearchBase,
	}

	target := server.resolveUpstreamTarget("/v1/chat/completions", []byte(`{"web_search_options":{}}`))
	if target.name != "websearch" {
		t.Fatalf("expected websearch target, got %q", target.name)
	}
	if target.baseURL.Host != "search.example.com" {
		t.Fatalf("expected search host, got %q", target.baseURL.Host)
	}

	target = server.resolveUpstreamTarget("/v1/chat/completions", []byte(`{"messages":[]}`))
	if target.name != "enclave" {
		t.Fatalf("expected enclave target, got %q", target.name)
	}
	if target.baseURL.Host != "enclave.example.com" {
		t.Fatalf("expected enclave host, got %q", target.baseURL.Host)
	}
}

func TestDoWithResponseStartTimeoutTimesOut(t *testing.T) {
	client := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			<-req.Context().Done()
			return nil, req.Context().Err()
		}),
	}

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, "https://example.com", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}

	start := time.Now()
	_, err = doWithResponseStartTimeout(client, req, 10*time.Millisecond)
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if err != context.DeadlineExceeded {
		t.Fatalf("expected context deadline exceeded, got %v", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("expected timeout quickly, took %s", elapsed)
	}
}

func TestDoWithResponseStartTimeoutAllowsLongBodyReads(t *testing.T) {
	reader, writer := io.Pipe()

	client := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{},
				Body:       reader,
			}, nil
		}),
	}

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, "https://example.com", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}

	resp, err := doWithResponseStartTimeout(client, req, 10*time.Millisecond)
	if err != nil {
		t.Fatalf("expected response, got %v", err)
	}
	defer resp.Body.Close()

	go func() {
		time.Sleep(25 * time.Millisecond)
		_, _ = writer.Write([]byte("ok"))
		_ = writer.Close()
	}()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("failed to read body: %v", err)
	}
	if string(body) != "ok" {
		t.Fatalf("expected body %q, got %q", "ok", string(body))
	}
}
