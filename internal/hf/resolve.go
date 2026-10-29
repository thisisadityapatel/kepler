package hf

import (
	"context"
	"errors"
	"fmt"
	"path"
	"strings"
)

// ModelProfile is a fully pinned model in one format: exactly the files to download
// and what preflight needs to know about them.
type ModelProfile struct {
	Repo        string   `json:"repo_id"`
	Revision    string   `json:"revision"`
	Format      string   `json:"format"`
	Quant       string   `json:"quant"`
	Files       []File   `json:"files"`
	EntryFile   string   `json:"entry_file,omitempty"` // GGUF shard 1; "" = model directory (MLX)
	SizeBytes   int64    `json:"size_bytes"`
	Arch        ArchInfo `json:"arch"`
	Vision      bool     `json:"vision"` // repo ships a vision projector (not loaded for text benchmarks)
	Discovered  bool     `json:"discovered"`
	RequestedAs string   `json:"requested_as"`
	BaseModel   string   `json:"base_model,omitempty"`
}

// Resolution is a repo resolved to a pinned commit, with every usable variant.
type Resolution struct {
	Ref       Ref
	Info      ModelInfo
	Format    string
	BaseModel string
	Files     []File
	Variants  []Variant // GGUF only, in preference order (see RankVariants)
	MMProj    []File
}

// ErrNoUsableFormat means the repo has no GGUF or MLX weights.
var ErrNoUsableFormat = errors.New("no GGUF or MLX weights")

// Resolve fetches metadata and the pinned file list for ref.
func Resolve(ctx context.Context, c *Client, ref Ref) (*Resolution, error) {
	info, err := c.Info(ctx, ref.Repo, ref.Revision)
	if err != nil {
		if errors.Is(err, ErrGated) {
			return nil, fmt.Errorf("%w (accept the license on huggingface.co/%s and set HF_TOKEN)", err, ref.Repo)
		}
		return nil, err
	}
	files, err := c.Files(ctx, ref.Repo, info.SHA)
	if err != nil {
		return nil, err
	}
	res := &Resolution{Ref: ref, Info: info, Files: files, Format: Classify(info, files), BaseModel: BaseModel(info)}
	switch res.Format {
	case FormatGGUF:
		vs, mm := GGUFVariants(files)
		res.Variants, res.MMProj = RankVariants(vs), mm
		if len(res.Variants) == 0 {
			return nil, fmt.Errorf("%s contains only vision projector GGUF files", ref.Repo)
		}
	case FormatMLX:
	default:
		return res, fmt.Errorf("%s: %w. Kepler v1 benchmarks GGUF (llama.cpp, Ollama) and MLX repos; "+
			"look for a GGUF build of %s (e.g. search \"%s GGUF\" on huggingface.co)",
			ref.Repo, ErrNoUsableFormat, res.BaseModel, path.Base(res.BaseModel))
	}
	return res, nil
}

// Candidates returns the GGUF variants to try, honoring an explicit file or quant in
// the reference. Without either, all variants are returned in preference order and
// preflight picks the first that fits.
func (r *Resolution) Candidates() ([]Variant, error) {
	if r.Format != FormatGGUF {
		return nil, fmt.Errorf("%s is %s, not GGUF", r.Ref.Repo, r.Format)
	}
	if r.Ref.File != "" {
		for _, v := range r.Variants {
			for _, f := range v.Files {
				if f.Path == r.Ref.File {
					return []Variant{v}, nil
				}
			}
		}
		return nil, fmt.Errorf("%s has no file %q", r.Ref.Repo, r.Ref.File)
	}
	if r.Ref.Quant != "" {
		if v, ok := FindVariant(r.Variants, r.Ref.Quant); ok {
			return []Variant{v}, nil
		}
		return nil, fmt.Errorf("%s has no %s variant (available: %s)", r.Ref.Repo, r.Ref.Quant, strings.Join(r.QuantNames(), ", "))
	}
	return r.Variants, nil
}

// QuantNames lists the available GGUF quants in preference order.
func (r *Resolution) QuantNames() []string {
	var out []string
	for _, v := range r.Variants {
		q := v.Quant
		if q == "" {
			q = path.Base(v.Entry().Path)
		}
		out = append(out, q)
	}
	return out
}

// GGUFProfile builds the profile for one variant, reading architecture info from the
// first shard's header (only the header bytes are transferred).
func (r *Resolution) GGUFProfile(ctx context.Context, c *Client, v Variant) (ModelProfile, error) {
	p := r.baseProfile(FormatGGUF)
	p.Quant, p.Files, p.SizeBytes, p.EntryFile = v.Quant, v.Files, v.Size, v.Entry().Path
	p.Vision = len(r.MMProj) > 0
	rc, err := c.ReadRange(ctx, r.Ref.Repo, r.Info.SHA, v.Entry().Path, 0)
	if err != nil {
		return p, fmt.Errorf("read GGUF header: %w", err)
	}
	defer rc.Close()
	if p.Arch, err = ReadGGUFArch(rc); err != nil {
		return p, fmt.Errorf("%s: %w", v.Entry().Path, err)
	}
	return p, nil
}

// MLXProfile builds the profile for an MLX repo: every file needed to load the model.
func (r *Resolution) MLXProfile(ctx context.Context, c *Client) (ModelProfile, error) {
	p := r.baseProfile(FormatMLX)
	for _, f := range r.Files {
		if mlxNeeded(f.Path) {
			p.Files = append(p.Files, f)
			p.SizeBytes += f.Size
		}
	}
	var cfg struct {
		MLXConfig
		Quantization *struct {
			Bits int `json:"bits"`
		} `json:"quantization"`
	}
	if err := c.GetJSON(ctx, r.Ref.Repo, r.Info.SHA, "config.json", &cfg); err != nil {
		return p, fmt.Errorf("read config.json: %w", err)
	}
	p.Arch = cfg.MLXConfig.Arch()
	p.Quant = "bf16"
	if cfg.Quantization != nil && cfg.Quantization.Bits > 0 {
		p.Quant = fmt.Sprintf("%dbit", cfg.Quantization.Bits)
	}
	return p, nil
}

func (r *Resolution) baseProfile(format string) ModelProfile {
	return ModelProfile{Repo: r.Ref.Repo, Revision: r.Info.SHA, Format: format, RequestedAs: r.Ref.String(), BaseModel: r.BaseModel}
}

// mlxNeeded filters out docs, images, and weight formats MLX does not load.
func mlxNeeded(p string) bool {
	lower := strings.ToLower(p)
	base := path.Base(lower)
	if strings.HasPrefix(base, ".") || strings.HasPrefix(base, "license") || strings.HasPrefix(base, "readme") {
		return false
	}
	for _, ext := range []string{".md", ".png", ".jpg", ".jpeg", ".gif", ".gguf", ".bin", ".pth", ".pt", ".onnx", ".h5", ".msgpack", ".ckpt"} {
		if strings.HasSuffix(lower, ext) {
			return false
		}
	}
	return true
}
