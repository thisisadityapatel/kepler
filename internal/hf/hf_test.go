package hf

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/thisisadityapatel/kepler/internal/hf/hftest"
)

func TestParseRef(t *testing.T) {
	cases := map[string]Ref{
		"bartowski/SmolLM2-135M-Instruct-GGUF":                          {Repo: "bartowski/SmolLM2-135M-Instruct-GGUF"},
		"Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m":                        {Repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF", Quant: "Q4_K_M"},
		"Qwen/Qwen2.5-0.5B-Instruct-GGUF@abc123":                        {Repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF", Revision: "abc123"},
		"https://huggingface.co/unsloth/gemma-3-4b-it-GGUF":             {Repo: "unsloth/gemma-3-4b-it-GGUF"},
		"https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/tree/main":   {Repo: "unsloth/gemma-3-4b-it-GGUF"},
		"hf.co/unsloth/gemma-3-4b-it-GGUF/tree/v2":                      {Repo: "unsloth/gemma-3-4b-it-GGUF", Revision: "v2"},
		"huggingface.co/a/b/blob/main/sub/Model-Q8_0.gguf":              {Repo: "a/b", File: "sub/Model-Q8_0.gguf"},
		"https://huggingface.co/a/b/resolve/deadbeef/x-Q4_K_M.gguf?d=1": {Repo: "a/b", Revision: "deadbeef", File: "x-Q4_K_M.gguf"},
	}
	for in, want := range cases {
		got, err := ParseRef(in)
		if err != nil || got != want {
			t.Errorf("ParseRef(%q) = %+v, %v; want %+v", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "llama3", "https://github.com/a/b", "https://huggingface.co/datasets/a/b", "a/b/c", "https://huggingface.co/a/b/blob/main/README.md"} {
		if _, err := ParseRef(bad); err == nil {
			t.Errorf("ParseRef(%q) should fail", bad)
		}
	}
}

func TestParseQuant(t *testing.T) {
	cases := map[string]string{
		"qwen2.5-0.5b-instruct-q4_k_m.gguf":           "Q4_K_M",
		"gemma-3-4b-it-UD-Q4_K_XL.gguf":               "UD-Q4_K_XL",
		"gemma-3-4b-it-IQ4_XS.gguf":                   "IQ4_XS",
		"gemma-3-4b-it-BF16.gguf":                     "BF16",
		"qwen2.5-0.5b-instruct-fp16.gguf":             "F16",
		"BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf": "BF16",
		"Q4_K_M/Model-00001-of-00003.gguf":            "Q4_K_M",
		"Meta-Llama-3.1-8B-Instruct-Q8_0.gguf":        "Q8_0",
		"model.gguf":                                  "",
	}
	for in, want := range cases {
		if got := ParseQuant(in); got != want {
			t.Errorf("ParseQuant(%q) = %q, want %q", in, got, want)
		}
	}
}

// fixtureHub serves recorded Hugging Face API responses from testdata/hf.
func fixtureHub(t *testing.T) *httptest.Server {
	t.Helper()
	root := filepath.Join("..", "..", "testdata", "hf")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimPrefix(r.URL.Path, "/api/models/")
		parts := strings.Split(p, "/")
		if len(parts) < 2 {
			http.NotFound(w, r)
			return
		}
		dir := filepath.Join(root, parts[0]+"__"+parts[1])
		file := "info.json"
		switch {
		case len(parts) >= 3 && parts[2] == "tree":
			file = "tree.json"
		case strings.Contains(r.URL.Path, "/resolve/") && strings.HasSuffix(r.URL.Path, "config.json"):
			dir = filepath.Join(root, strings.Split(strings.TrimPrefix(r.URL.Path, "/"), "/")[0]+"__"+strings.Split(strings.TrimPrefix(r.URL.Path, "/"), "/")[1])
			file = "config.json"
		}
		b, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil {
			http.NotFound(w, r)
			return
		}
		w.Write(b)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestResolveRecordedRepos(t *testing.T) {
	ctx := context.Background()
	c := NewClient(fixtureHub(t).URL)

	res, err := Resolve(ctx, c, Ref{Repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Format != FormatGGUF || res.Variants[0].Quant != "Q4_K_M" || res.BaseModel != "Qwen/Qwen2.5-0.5B-Instruct" {
		t.Fatalf("qwen: format=%s first=%s base=%s", res.Format, res.Variants[0].Quant, res.BaseModel)
	}
	if last := res.Variants[len(res.Variants)-1]; last.Quant != "F16" {
		t.Errorf("full precision should rank last, got %s", last.Quant)
	}

	res, err = Resolve(ctx, c, Ref{Repo: "unsloth/gemma-3-4b-it-GGUF", Quant: "UD-Q4_K_XL"})
	if err != nil {
		t.Fatal(err)
	}
	cands, err := res.Candidates()
	if err != nil || len(cands) != 1 || cands[0].Quant != "UD-Q4_K_XL" {
		t.Fatalf("gemma candidates = %+v %v", cands, err)
	}
	if len(res.MMProj) != 3 || PickMMProj(res.MMProj).Path != "mmproj-F16.gguf" {
		t.Errorf("mmproj = %+v", res.MMProj)
	}

	res, err = Resolve(ctx, c, Ref{Repo: "unsloth/Qwen3-30B-A3B-GGUF", Quant: "BF16"})
	if err != nil {
		t.Fatal(err)
	}
	cands, _ = res.Candidates()
	if len(cands) != 1 || len(cands[0].Files) != 2 || !strings.HasSuffix(cands[0].Entry().Path, "-00001-of-00002.gguf") {
		t.Fatalf("sharded variant = %+v", cands)
	}
	if cands[0].Size != cands[0].Files[0].Size+cands[0].Files[1].Size {
		t.Error("shard sizes not summed")
	}

	if _, err := (&Resolution{Ref: Ref{Repo: "x/y", Quant: "Q9_K"}, Format: FormatGGUF, Variants: res.Variants}).Candidates(); err == nil || !strings.Contains(err.Error(), "available:") {
		t.Errorf("missing quant error = %v", err)
	}

	res, err = Resolve(ctx, c, Ref{Repo: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"})
	if err != nil || res.Format != FormatMLX {
		t.Fatalf("mlx: %v %v", res, err)
	}
	p, err := res.MLXProfile(ctx, c)
	if err != nil {
		t.Fatal(err)
	}
	if p.Quant != "4bit" || p.Arch.Layers != 24 || p.Arch.KVHeads != 2 || p.Arch.HeadDim != 64 {
		t.Errorf("mlx profile = quant %s arch %+v", p.Quant, p.Arch)
	}
	for _, f := range p.Files {
		if strings.HasSuffix(f.Path, ".md") || f.Path == ".gitattributes" {
			t.Errorf("doc file included: %s", f.Path)
		}
	}

	_, err = Resolve(ctx, c, Ref{Repo: "Qwen/Qwen2.5-0.5B-Instruct"})
	if !errors.Is(err, ErrNoUsableFormat) || !strings.Contains(err.Error(), "GGUF build of Qwen/Qwen2.5-0.5B") {
		t.Errorf("safetensors-only error = %v", err)
	}
	if _, err := Resolve(ctx, c, Ref{Repo: "nobody/missing"}); !errors.Is(err, ErrNotFound) {
		t.Errorf("missing repo error = %v", err)
	}
}

func TestReadGGUFArch(t *testing.T) {
	b := hftest.GGUF("qwen2", 24, 14, 2, 896, 32768, 0)
	a, err := ReadGGUFArch(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	if a.Architecture != "qwen2" || a.Layers != 24 || a.KVHeads != 2 || a.HeadDim != 64 || a.ContextMax != 32768 {
		t.Fatalf("arch = %+v", a)
	}
	if _, err := ReadGGUFArch(strings.NewReader("not a gguf file at all")); err == nil {
		t.Fatal("expected error for non-GGUF")
	}
}

func TestGGUFProfileReadsHeaderRemotely(t *testing.T) {
	model := hftest.GGUF("llama", 16, 32, 8, 2048, 131072, 4<<20)
	hub := hftest.NewHub(&hftest.Repo{ID: "org/Tiny-GGUF", SHA: strings.Repeat("c", 40),
		Files: map[string][]byte{"Tiny-Q4_K_M.gguf": model, "Tiny-Q8_0.gguf": model[:len(model)-1]}})
	defer hub.Close()
	c := NewClient(hub.URL)
	res, err := Resolve(context.Background(), c, Ref{Repo: "org/Tiny-GGUF"})
	if err != nil {
		t.Fatal(err)
	}
	p, err := res.GGUFProfile(context.Background(), c, res.Variants[0])
	if err != nil {
		t.Fatal(err)
	}
	if p.Quant != "Q4_K_M" || p.Arch.Layers != 16 || p.Arch.HeadDim != 64 || p.Revision != strings.Repeat("c", 40) || p.Files[0].SHA256 == "" {
		t.Fatalf("profile = %+v", p)
	}
	var round ModelProfile
	b, _ := json.Marshal(p)
	if err := json.Unmarshal(b, &round); err != nil || round.Arch != p.Arch {
		t.Fatalf("json round trip: %v", err)
	}
}
