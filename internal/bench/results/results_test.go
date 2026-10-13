package results

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/thisisadityapatel/kepler/internal/hf"
)

func sample(engine string, decode float64, flags []string) *Result {
	r := &Result{
		SchemaVersion: SchemaVersion, Timestamp: time.Date(2026, 10, 12, 15, 8, 30, 0, time.Local), RunID: "r1",
		Status: StatusComplete, Model: hf.ModelProfile{Repo: "org/Model-GGUF", Format: "gguf", Quant: "Q4_K_M"},
		Engine: EngineInfo{Name: engine}, Config: Config{Preset: "quick"},
		Load:          Load{LoadS: 1.5, PeakMemBytes: 1 << 30},
		Comparability: Comparability{Status: ComparabilityFull},
	}
	s := Scenario{Name: "chat", Kind: "chat", MaxTokens: 256, Flags: flags, Iterations: []Iteration{
		{DecodeTokPerS: decode - 1, TTFTMs: 50, PeakMemBytes: 2 << 30},
		{DecodeTokPerS: decode, TTFTMs: 40, PeakMemBytes: 2 << 30},
		{DecodeTokPerS: decode + 1, TTFTMs: 45, PeakMemBytes: 3 << 30},
	}}
	s.Summarize()
	r.Scenarios = []Scenario{s}
	return r
}

func TestWriteReadAndFilename(t *testing.T) {
	dir := t.TempDir()
	r := sample("llamacpp", 40, nil)
	if got := Filename(r); got != "2026-10-12T150830_org-Model-GGUF_llamacpp_quick.json" {
		t.Errorf("filename = %s", got)
	}
	p1, err := Write(dir, r)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := Write(dir, r)
	if err != nil || p1 == p2 || !strings.HasSuffix(p2, "_2.json") {
		t.Fatalf("second write must not overwrite: %s %s %v", p1, p2, err)
	}
	back, err := Read(p1)
	if err != nil {
		t.Fatal(err)
	}
	if back.Scenarios[0].Summary.DecodeTokPerS.Median != 40 || back.PeakMem() != 3<<30 {
		t.Fatalf("round trip = %+v", back.Scenarios[0].Summary)
	}
	os.WriteFile(filepath.Join(dir, "old.json"), []byte(`{"schema_version":1}`), 0o644)
	if _, err := Read(filepath.Join(dir, "old.json")); err == nil {
		t.Error("schema v1 accepted")
	}
}

func TestComparisonAndReport(t *testing.T) {
	a, b := sample("llamacpp", 40, nil), sample("ollama", 35, []string{"stopped early"})
	c := BuildComparison("c1", "org/Model-GGUF", "quick", SystemInfo{}, []*Result{a, b}, []string{"/x/a.json", "/x/b.json"})
	if got := c.Ranking["decode_tok_per_s"]; len(got) != 2 || got[0] != "llamacpp" {
		t.Errorf("ranking = %v", c.Ranking)
	}
	if c.Entries[1].ResultFile != "b.json" || len(c.Notes) != 0 {
		t.Errorf("entries/notes = %+v %v", c.Entries, c.Notes)
	}
	b.Model.Quant = "Q8_0"
	if c := BuildComparison("c1", "", "quick", SystemInfo{}, []*Result{a, b}, []string{"a", "b"}); len(c.Notes) != 1 {
		t.Error("mixed quants must add an approximate-comparison note")
	}

	tab := Report([]*Result{a, b})
	if tab.Header[4] != "decode tok/s" || tab.Rows[0][4] != "40.0" || tab.Rows[1][4] != "35.0 ~" {
		t.Fatalf("report = %+v", tab)
	}
	if !tab.Best[0][4] || tab.Best[1][4] {
		t.Errorf("best marks = %v", tab.Best)
	}
	if len(tab.Legend) != 2 {
		t.Errorf("legend = %v", tab.Legend)
	}
}
