package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestStreamParsesTokensUsageAndExtra(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		for _, tok := range []string{"Hel", "lo", "!"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", tok)
			fl.Flush()
			time.Sleep(5 * time.Millisecond)
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"timings\":{\"predicted_per_second\":42.0}}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	seed := 42
	res, err := New(srv.URL+"/v1/").Stream(context.Background(), ChatRequest{
		Model: "m", Messages: []Message{{Role: "user", Content: "hi"}}, MaxTokens: 3, Seed: &seed,
		Extra: map[string]any{"ignore_eos": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "Hello!" || res.Events() != 3 || res.FinishReason != "length" {
		t.Fatalf("res = %+v", res)
	}
	if res.Usage == nil || res.Usage.CompletionTokens != 3 || len(res.Timings) == 0 {
		t.Fatalf("usage/timings = %+v %s", res.Usage, res.Timings)
	}
	if !res.TokenTimes[2].After(res.TokenTimes[0]) || res.Closed.Before(res.TokenTimes[2]) {
		t.Fatal("timestamps out of order")
	}
	if got["ignore_eos"] != true || got["stream"] != true || got["seed"] != float64(42) {
		t.Fatalf("request body = %v", got)
	}
}

func TestStreamHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not found", http.StatusNotFound)
	}))
	defer srv.Close()
	if _, err := New(srv.URL).Stream(context.Background(), ChatRequest{Model: "x"}); err == nil {
		t.Fatal("expected error")
	}
}
