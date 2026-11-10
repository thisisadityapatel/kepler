package runner

import (
	"sync"
	"time"

	"github.com/thisisadityapatel/kepler/internal/proc"
)

// Sampler polls the engine's process-group memory in the background. The value is
// phys_footprint (anonymous + GPU memory) plus resident pages of memory-mapped model
// files under mappedPrefix, which phys_footprint leaves out.
type Sampler struct {
	pgid       int
	interval   time.Duration
	read       func(pgid int) (int64, error)
	readMapped func(pgid int) (int64, error)
	mapped     int64
	samples    int

	mu       sync.Mutex
	current  int64
	peak     int64
	window   int64
	stop     chan struct{}
	done     chan struct{}
	failures int
}

// NewSampler returns a sampler for pgid; call Start. mappedPrefix ("" = none) selects
// which mapped files count as model memory.
func NewSampler(pgid int, interval time.Duration, mappedPrefix string) *Sampler {
	s := &Sampler{pgid: pgid, interval: interval, read: proc.GroupFootprint, stop: make(chan struct{}), done: make(chan struct{})}
	if mappedPrefix != "" {
		s.readMapped = func(pgid int) (int64, error) { return proc.GroupMappedResident(pgid, mappedPrefix) }
	}
	return s
}

// mappedEvery is how many samples pass between (slower) mapped-file scans.
const mappedEvery = 10

// Start begins sampling.
func (s *Sampler) Start() {
	go func() {
		defer close(s.done)
		t := time.NewTicker(s.interval)
		defer t.Stop()
		for {
			s.Sample()
			select {
			case <-s.stop:
				return
			case <-t.C:
			}
		}
	}()
}

// Sample takes one reading now and returns it.
func (s *Sampler) Sample() int64 {
	return s.sample(false)
}

func (s *Sampler) sample(forceMapped bool) int64 {
	v, err := s.read(s.pgid)
	s.mu.Lock()
	scanMapped := s.readMapped != nil && (forceMapped || s.samples%mappedEvery == 0)
	s.mu.Unlock()
	var mapped int64 = -1
	if scanMapped {
		if m, merr := s.readMapped(s.pgid); merr == nil {
			mapped = m
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.samples++
	if mapped >= 0 {
		s.mapped = mapped
	}
	if err != nil {
		s.failures++
		return s.current
	}
	v += s.mapped
	s.current = v
	if v > s.peak {
		s.peak = v
	}
	if v > s.window {
		s.window = v
	}
	return v
}

// SampleFull takes a reading that includes a fresh mapped-file scan.
func (s *Sampler) SampleFull() int64 { return s.sample(true) }

// Mapped is the latest mapped-model-file component.
func (s *Sampler) Mapped() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.mapped
}

// ResetWindow starts a new per-iteration peak window.
func (s *Sampler) ResetWindow() {
	s.mu.Lock()
	s.window = s.current
	s.mu.Unlock()
}

// WindowPeak is the peak since the last ResetWindow (taking a final reading first).
func (s *Sampler) WindowPeak() int64 {
	s.SampleFull()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.window
}

// Peak is the peak since Start.
func (s *Sampler) Peak() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.peak
}

// Working reports whether any sample has succeeded.
func (s *Sampler) Working() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.peak > 0
}

// Stop ends sampling and waits for the goroutine.
func (s *Sampler) Stop() {
	select {
	case <-s.stop:
	default:
		close(s.stop)
	}
	<-s.done
}
