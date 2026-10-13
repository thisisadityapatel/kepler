// Package events is the in-process event stream. Shared-core packages publish typed
// events; the CLI (and later a daemon) subscribe. Publishing never blocks.
package events

import (
	"sync"
	"time"
)

// Event types.
const (
	InstallStarted   = "install.started"
	InstallStep      = "install.step"
	InstallFinished  = "install.finished"
	DownloadProgress = "download.progress"
	EngineLaunched   = "engine.launched"
	EngineReady      = "engine.ready"
	EngineStopped    = "engine.stopped"
	HealthFinished   = "health.finished"
	Warning          = "warning"
	BenchPhase       = "bench.phase"     // Message: what the engine is doing
	BenchIteration   = "bench.iteration" // Message: scenario; Done/Total: iteration
	BenchEngineDone  = "bench.engine"    // Message: one-line summary; Err on failure
)

// Event is one published event.
type Event struct {
	Type    string
	Time    time.Time
	Engine  string
	Message string
	Done    int64 // progress events
	Total   int64
	Err     error
}

// Bus fans events out to subscribers.
type Bus struct {
	mu   sync.Mutex
	subs []func(Event)
}

// Subscribe registers fn; it is called synchronously in publish order and must be fast.
func (b *Bus) Subscribe(fn func(Event)) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.subs = append(b.subs, fn)
}

// Publish sends e to every subscriber. A nil Bus discards events.
func (b *Bus) Publish(e Event) {
	if b == nil {
		return
	}
	if e.Time.IsZero() {
		e.Time = time.Now()
	}
	b.mu.Lock()
	subs := append([]func(Event){}, b.subs...)
	b.mu.Unlock()
	for _, fn := range subs {
		fn(e)
	}
}
