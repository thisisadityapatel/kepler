// Package hftest provides a fake Hugging Face Hub and synthetic GGUF files for tests.
package hftest

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
)

// GGUF builds a minimal valid GGUF v3 header with llama-style metadata followed by
// padding up to size bytes (so downloads have realistic length).
func GGUF(arch string, layers, heads, kvHeads, embd, ctx int, size int) []byte {
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	kvs := []struct {
		key string
		val uint32
	}{
		{arch + ".block_count", uint32(layers)},
		{arch + ".attention.head_count", uint32(heads)},
		{arch + ".attention.head_count_kv", uint32(kvHeads)},
		{arch + ".embedding_length", uint32(embd)},
		{arch + ".context_length", uint32(ctx)},
	}
	w(uint32(0x46554747))
	w(uint32(3))
	w(uint64(0))
	w(uint64(len(kvs) + 2))
	str("general.architecture")
	w(uint32(8))
	str(arch)
	for _, kv := range kvs {
		str(kv.key)
		w(uint32(4))
		w(kv.val)
	}
	str("tokenizer.ggml.tokens") // array of strings, should be skipped/never needed
	w(uint32(9))
	w(uint32(8))
	w(uint64(2))
	str("a")
	str("b")
	for b.Len() < size {
		b.WriteByte(0)
	}
	return b.Bytes()
}

// Repo is one fake repository.
type Repo struct {
	ID    string
	SHA   string
	Tags  []string
	Files map[string][]byte // path → content (LFS for *.gguf and *.safetensors)
	Gated bool
}

// Hub is a fake Hugging Face Hub serving /api/models and /resolve.
type Hub struct {
	*httptest.Server
	mu        sync.Mutex
	repos     map[string]*Repo
	Downloads map[string]int // path → number of full/partial GETs
}

// NewHub starts a hub serving repos.
func NewHub(repos ...*Repo) *Hub {
	h := &Hub{repos: map[string]*Repo{}, Downloads: map[string]int{}}
	for _, r := range repos {
		h.repos[r.ID] = r
	}
	h.Server = httptest.NewServer(http.HandlerFunc(h.serve))
	return h
}

func isLFS(p string) bool {
	return strings.HasSuffix(p, ".gguf") || strings.HasSuffix(p, ".safetensors")
}

func (h *Hub) serve(w http.ResponseWriter, r *http.Request) {
	p := strings.TrimPrefix(r.URL.Path, "/")
	if strings.HasPrefix(p, "api/models/") {
		h.api(w, strings.TrimPrefix(p, "api/models/"))
		return
	}
	// {org}/{name}/resolve/{sha}/{path...}
	parts := strings.SplitN(p, "/", 5)
	if len(parts) < 5 || parts[2] != "resolve" {
		http.NotFound(w, r)
		return
	}
	repo := h.repos[parts[0]+"/"+parts[1]]
	if repo == nil {
		http.NotFound(w, r)
		return
	}
	if repo.Gated && r.Header.Get("Authorization") == "" {
		http.Error(w, "gated", http.StatusUnauthorized)
		return
	}
	body, ok := repo.Files[parts[4]]
	if !ok {
		http.NotFound(w, r)
		return
	}
	h.mu.Lock()
	h.Downloads[parts[4]]++
	h.mu.Unlock()
	start := 0
	if rg := r.Header.Get("Range"); strings.HasPrefix(rg, "bytes=") {
		start, _ = strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
		if start > 0 {
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(body)-1, len(body)))
			w.WriteHeader(http.StatusPartialContent)
		}
	}
	_, _ = w.Write(body[start:])
}

func (h *Hub) api(w http.ResponseWriter, rest string) {
	parts := strings.Split(rest, "/")
	if len(parts) < 2 {
		http.NotFound(w, nil)
		return
	}
	repo := h.repos[parts[0]+"/"+parts[1]]
	if repo == nil {
		http.Error(w, `{"error":"Repository not found"}`, http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if len(parts) >= 4 && parts[2] == "tree" {
		var out []map[string]any
		for path, body := range repo.Files {
			e := map[string]any{"type": "file", "path": path, "size": len(body), "oid": gitOID(body)}
			if isLFS(path) {
				sum := sha256.Sum256(body)
				e["lfs"] = map[string]any{"oid": hex.EncodeToString(sum[:]), "size": len(body)}
			}
			out = append(out, e)
		}
		_ = json.NewEncoder(w).Encode(out)
		return
	}
	lib := ""
	for _, t := range repo.Tags {
		if t == "mlx" {
			lib = "mlx"
		}
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"id": repo.ID, "sha": repo.SHA, "tags": repo.Tags, "gated": repo.Gated, "library_name": lib})
}

func gitOID(b []byte) string {
	h := sha1New()
	fmt.Fprintf(h, "blob %d\x00", len(b))
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
}
