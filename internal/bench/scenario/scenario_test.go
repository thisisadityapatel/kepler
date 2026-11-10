package scenario

import (
	"strings"
	"testing"
)

func TestPresetsAndCtx(t *testing.T) {
	q, err := Build("quick", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if q.Iterations != 3 || len(q.Scenarios) != 2 || q.Scenarios[0].MaxTokens != 256 {
		t.Fatalf("quick = %+v", q)
	}
	if ctx := Ctx(q.Scenarios); ctx%512 != 0 || ctx < Catalog["prefill-2k"].Need() {
		t.Errorf("quick ctx = %d", ctx)
	}
	s, _ := Build("standard", nil, 0)
	if ctx := Ctx(s.Scenarios); ctx < 16384+128 || ctx > 20480 {
		t.Errorf("standard ctx = %d", ctx)
	}
	custom, err := Build("standard", []string{"depth-4k", "chat"}, 7)
	if err != nil || custom.Scenarios[0].Name != "chat" || custom.Iterations != 7 {
		t.Fatalf("custom = %+v %v", custom, err)
	}
	if _, err := Build("nope", nil, 0); err == nil {
		t.Error("unknown preset accepted")
	}
	if _, err := Build("quick", []string{"chat", "bogus"}, 0); err == nil {
		t.Error("unknown scenario accepted")
	}
}

func TestFitDropsScenariosOverContext(t *testing.T) {
	f, _ := Build("full", nil, 0)
	kept, skipped := Fit(f.Scenarios, 8192)
	for _, s := range kept {
		if s.Need() > 8192 {
			t.Errorf("kept %s needing %d", s.Name, s.Need())
		}
	}
	if len(skipped) != 3 { // prefill-8k, depth-16k, depth-32k
		t.Errorf("skipped = %+v", skipped)
	}
}

func TestPromptDeterministicSizedAndUnique(t *testing.T) {
	s := Catalog["prefill-2k"]
	a := Prompt(s, "n1", 4)
	b := Prompt(s, "n2", 4)
	if strings.TrimPrefix(a, "[session n1]\n") != strings.TrimPrefix(b, "[session n2]\n") {
		t.Error("prompt body must be identical across iterations and engines")
	}
	if a[:20] == b[:20] {
		t.Error("prefix must differ so prompt caches cannot help")
	}
	if n := len(a); n < 2048*4-200 || n > 2048*4+400 {
		t.Errorf("prompt length %d not near %d chars", n, 2048*4)
	}
	if len(Prompt(s, "n", 2)) >= len(a) {
		t.Error("calibration must shrink prompts for tokenizers with fewer chars/token")
	}
}
