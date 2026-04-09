package main

import (
	"context"
	"io"
	"net/http"
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
