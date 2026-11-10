// Package fakeengine is a deterministic OpenAI-compatible streaming server used by
// tests in place of a real engine. Tests re-exec their own binary with
// KEPLER_FAKE_ENGINE=1 and call MaybeServe from TestMain.
package fakeengine

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"sync/atomic"
	"time"
)

// EnvFlag, when set to "1", turns a test binary into a fake engine.
const EnvFlag = "KEPLER_FAKE_ENGINE"

// Options control the fake's behavior (also settable via env for re-exec).
type Options struct {
	TokenDelay   time.Duration // between token events
	StartupDelay time.Duration // before /health returns 200
	ExitAtStart  bool          // simulate a crash on startup
	StopAt       int           // if > 0, never emit more than this many tokens (no ignore_eos)
	NoUsage      bool          // omit usage from the stream
	DieAfter     int           // if > 0, exit the process after this many chat requests
}

func optionsFromEnv() Options {
	o := Options{TokenDelay: 2 * time.Millisecond}
	if v, err := time.ParseDuration(os.Getenv("FAKE_TOKEN_DELAY")); err == nil {
		o.TokenDelay = v
	}
	if v, err := time.ParseDuration(os.Getenv("FAKE_STARTUP_DELAY")); err == nil {
		o.StartupDelay = v
	}
	o.ExitAtStart = os.Getenv("FAKE_EXIT_AT_START") == "1"
	o.StopAt, _ = strconv.Atoi(os.Getenv("FAKE_STOP_AT"))
	o.NoUsage = os.Getenv("FAKE_NO_USAGE") == "1"
	o.DieAfter, _ = strconv.Atoi(os.Getenv("FAKE_DIE_AFTER"))
	return o
}

// MaybeServe runs the fake engine and exits if EnvFlag is set; otherwise it returns.
// Recognized args: --port N (others ignored, so real engine arg lists work).
func MaybeServe() {
	if os.Getenv(EnvFlag) != "1" {
		return
	}
	fs := flag.NewFlagSet("fake", flag.ContinueOnError)
	port := fs.Int("port", 0, "")
	var args []string
	for i, a := range os.Args {
		if a == "--" {
			args = os.Args[i+1:]
			break
		}
	}
	for i := 0; i < len(args); i++ { // pick out --port N, ignore everything else
		if args[i] == "--port" && i+1 < len(args) {
			_ = fs.Parse([]string{"--port", args[i+1]})
		}
	}
	if err := Serve(*port, optionsFromEnv()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	os.Exit(0)
}

// Serve blocks serving the fake API on 127.0.0.1:port.
func Serve(port int, o Options) error {
	if o.ExitAtStart {
		return fmt.Errorf("fake engine: simulated startup crash")
	}
	fmt.Println("fake engine starting on port", port)
	readyAt := time.Now().Add(o.StartupDelay)
	mux := http.NewServeMux()
	health := func(w http.ResponseWriter, r *http.Request) {
		if time.Now().Before(readyAt) {
			http.Error(w, "loading", http.StatusServiceUnavailable)
			return
		}
		w.Write([]byte(`{"status":"ok"}`))
	}
	mux.HandleFunc("/health", health)
	mux.HandleFunc("/v1/models", health)
	var requests atomic.Int64
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			MaxTokens int    `json:"max_tokens"`
			Model     string `json:"model"`
			Messages  []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if n := requests.Add(1); o.DieAfter > 0 && n > int64(o.DieAfter) {
			os.Exit(3)
		}
		if req.MaxTokens <= 0 {
			req.MaxTokens = 16
		}
		promptChars := 0
		for _, m := range req.Messages {
			promptChars += len(m.Content)
		}
		promptTokens := promptChars/4 + 10 // 4 chars/token plus a chat-template overhead
		if o.StopAt > 0 && req.MaxTokens > o.StopAt {
			req.MaxTokens = o.StopAt
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fl, _ := w.(http.Flusher)
		for i := 0; i < req.MaxTokens; i++ {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", "t"+strconv.Itoa(i)+" ")
			if fl != nil {
				fl.Flush()
			}
			time.Sleep(o.TokenDelay)
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n")
		if !o.NoUsage {
			fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d}}\n\n", promptTokens, req.MaxTokens)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	return http.ListenAndServe("127.0.0.1:"+strconv.Itoa(port), mux)
}
