// Package openai is a minimal streaming client for OpenAI-compatible
// /v1/chat/completions endpoints. It timestamps every token event with a monotonic
// clock so callers can derive TTFT, decode rate, and inter-token latency.
package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Message is one chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest is the request body. Extra holds engine-specific fields merged at the top
// level (e.g. ignore_eos, cache_prompt).
type ChatRequest struct {
	Model       string
	Messages    []Message
	MaxTokens   int
	Temperature float64
	Seed        *int
	Extra       map[string]any
}

// MarshalJSON flattens Extra into the top-level object.
func (r ChatRequest) MarshalJSON() ([]byte, error) {
	m := map[string]any{}
	for k, v := range r.Extra {
		m[k] = v
	}
	m["model"] = r.Model
	m["messages"] = r.Messages
	m["max_tokens"] = r.MaxTokens
	m["temperature"] = r.Temperature
	m["stream"] = true
	m["stream_options"] = map[string]any{"include_usage": true}
	if r.Seed != nil {
		m["seed"] = *r.Seed
	}
	return json.Marshal(m)
}

// Usage is the token accounting reported by the engine (zero if not reported).
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// StreamResult is everything observed during one streamed completion.
type StreamResult struct {
	Sent         time.Time       // request sent
	TokenTimes   []time.Time     // one per token event (content or reasoning delta)
	Closed       time.Time       // stream ended
	Text         string          // concatenated content (reasoning excluded)
	Usage        *Usage          // nil if the engine did not report usage
	Timings      json.RawMessage // engine-specific timings (llama-server), if present
	FinishReason string
}

// Events is the number of token events observed.
func (s StreamResult) Events() int { return len(s.TokenTimes) }

// Client talks to one engine endpoint.
type Client struct {
	BaseURL string // e.g. http://127.0.0.1:51234/v1
	HTTP    *http.Client
}

// New returns a client for baseURL.
func New(baseURL string) *Client {
	return &Client{BaseURL: strings.TrimRight(baseURL, "/"), HTTP: &http.Client{}}
}

type chunk struct {
	Choices []struct {
		Delta struct {
			Content          string `json:"content"`
			ReasoningContent string `json:"reasoning_content"`
			Reasoning        string `json:"reasoning"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage   *Usage          `json:"usage"`
	Timings json.RawMessage `json:"timings"`
}

// Stream sends req and consumes the SSE stream to completion.
func (c *Client) Stream(ctx context.Context, req ChatRequest) (StreamResult, error) {
	var res StreamResult
	body, err := json.Marshal(req)
	if err != nil {
		return res, err
	}
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return res, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")

	res.Sent = time.Now()
	resp, err := c.HTTP.Do(hreq)
	if err != nil {
		return res, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return res, fmt.Errorf("chat completion: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	var text strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		now := time.Now()
		var ch chunk
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			return res, fmt.Errorf("decode stream chunk: %w", err)
		}
		for _, choice := range ch.Choices {
			d := choice.Delta
			if d.Content != "" || d.ReasoningContent != "" || d.Reasoning != "" {
				res.TokenTimes = append(res.TokenTimes, now)
				text.WriteString(d.Content)
			}
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				res.FinishReason = *choice.FinishReason
			}
		}
		if ch.Usage != nil && (ch.Usage.CompletionTokens > 0 || ch.Usage.PromptTokens > 0) {
			res.Usage = ch.Usage
		}
		if len(ch.Timings) > 0 && string(ch.Timings) != "null" {
			res.Timings = ch.Timings
		}
	}
	res.Closed = time.Now()
	res.Text = text.String()
	if err := sc.Err(); err != nil {
		return res, fmt.Errorf("read stream: %w", err)
	}
	if len(res.TokenTimes) == 0 {
		return res, fmt.Errorf("stream ended without any tokens")
	}
	return res, nil
}
