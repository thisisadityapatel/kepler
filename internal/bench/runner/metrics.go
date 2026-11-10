package runner

import (
	"fmt"
	"time"

	"github.com/thisisadityapatel/kepler/internal/bench/results"
	"github.com/thisisadityapatel/kepler/internal/bench/stats"
	"github.com/thisisadityapatel/kepler/internal/openai"
)

// Observation records how an iteration's numbers were obtained, for comparability.
type Observation struct {
	UsageMissing bool // token counts came from stream events
	ITLApprox    bool // events != tokens (engine batches tokens per chunk)
	StoppedEarly bool // fewer than max_tokens generated
}

// Measure turns a stream into iteration metrics (DESIGN.md §9.5).
func Measure(res openai.StreamResult, maxTokens int, estPromptTokens int) (results.Iteration, Observation) {
	var it results.Iteration
	var obs Observation
	n := res.Events()
	it.TokenEvents = n
	it.FinishReason = res.FinishReason
	it.EngineReported = res.Timings

	if res.Usage != nil && res.Usage.CompletionTokens > 0 {
		it.GeneratedTokens = res.Usage.CompletionTokens
		it.PromptTokens = res.Usage.PromptTokens
	} else {
		it.GeneratedTokens = n
		obs.UsageMissing = true
	}
	if it.PromptTokens == 0 {
		it.PromptTokens = estPromptTokens
	}
	obs.ITLApprox = n != it.GeneratedTokens
	obs.StoppedEarly = it.GeneratedTokens < maxTokens

	ttft := res.TokenTimes[0].Sub(res.Sent)
	wall := res.Closed.Sub(res.Sent)
	it.TTFTMs = ms(ttft)
	it.WallS = wall.Seconds()
	if ttft > 0 {
		it.PrefillTokPerS = float64(it.PromptTokens) / ttft.Seconds()
	}
	if wall > 0 {
		it.E2ETokPerS = float64(it.GeneratedTokens) / wall.Seconds()
	}
	if n > 1 {
		span := res.TokenTimes[n-1].Sub(res.TokenTimes[0])
		if span > 0 && it.GeneratedTokens > 1 {
			it.DecodeTokPerS = float64(it.GeneratedTokens-1) / span.Seconds()
		}
		gaps := make([]float64, 0, n-1)
		for i := 1; i < n; i++ {
			gaps = append(gaps, ms(res.TokenTimes[i].Sub(res.TokenTimes[i-1])))
		}
		it.ITLP50Ms = stats.Percentile(gaps, 50)
		it.ITLP99Ms = stats.Percentile(gaps, 99)
	}
	return it, obs
}

func ms(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// scenarioFlags turns observations into human-readable flags for one scenario.
func scenarioFlags(obs []Observation, maxTokens int) []string {
	var flags []string
	early, usage, itl := 0, 0, 0
	for _, o := range obs {
		if o.StoppedEarly {
			early++
		}
		if o.UsageMissing {
			usage++
		}
		if o.ITLApprox {
			itl++
		}
	}
	if early > 0 {
		flags = append(flags, fmt.Sprintf("stopped before %d tokens in %d/%d iterations (engine does not support ignore_eos)", maxTokens, early, len(obs)))
	}
	if usage > 0 {
		flags = append(flags, fmt.Sprintf("token counts from stream events in %d/%d iterations (engine did not report usage)", usage, len(obs)))
	}
	if itl > 0 && usage == 0 {
		flags = append(flags, "inter-token latency approximate (engine sends several tokens per stream event)")
	}
	return flags
}
