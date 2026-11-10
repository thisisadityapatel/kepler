package hf

import (
	"path"
	"regexp"
	"sort"
	"strings"
)

// Formats Kepler understands.
const (
	FormatGGUF        = "gguf"
	FormatMLX         = "mlx"
	FormatSafetensors = "safetensors"
)

// Classify determines the repo's model format from its files and tags.
func Classify(info ModelInfo, files []File) string {
	hasGGUF, hasConfig, hasST := false, false, false
	for _, f := range files {
		switch {
		case strings.HasSuffix(strings.ToLower(f.Path), ".gguf"):
			hasGGUF = true
		case f.Path == "config.json":
			hasConfig = true
		case strings.HasSuffix(f.Path, ".safetensors"):
			hasST = true
		}
	}
	switch {
	case hasGGUF:
		return FormatGGUF
	case hasConfig && hasST && (hasTag(info.Tags, "mlx") || info.LibraryName == "mlx"):
		return FormatMLX
	case hasConfig && hasST:
		return FormatSafetensors
	}
	return ""
}

func hasTag(tags []string, t string) bool {
	for _, x := range tags {
		if x == t {
			return true
		}
	}
	return false
}

// BaseModel returns the upstream model id from card data or base_model tags.
func BaseModel(info ModelInfo) string {
	switch b := info.CardData.BaseModel.(type) {
	case string:
		return b
	case []any:
		if len(b) > 0 {
			if s, ok := b[0].(string); ok {
				return s
			}
		}
	}
	for _, t := range info.Tags {
		for _, p := range []string{"base_model:quantized:", "base_model:finetune:", "base_model:"} {
			if strings.HasPrefix(t, p) && !strings.Contains(strings.TrimPrefix(t, p), ":") {
				return strings.TrimPrefix(t, p)
			}
		}
	}
	return info.ID
}

// Variant is one quantization of a GGUF model: one file or a set of shards.
type Variant struct {
	Quant string // normalized, e.g. Q4_K_M, UD-Q4_K_XL, BF16; "" if unrecognized
	Files []File // shards in order; Files[0] is the entry file passed to engines
	Size  int64
}

// Entry is the file engines are pointed at (first shard).
func (v Variant) Entry() File { return v.Files[0] }

var (
	shardRe = regexp.MustCompile(`(?i)-(\d{5})-of-(\d{5})\.gguf$`)
	// Quant tokens as they appear in file names; matched case-insensitively.
	quantRe = regexp.MustCompile(`(?i)(?:^|[-_./])((?:UD-)?(?:I?Q[1-8](?:_[0-9A-Z]+)*|BF16|F16|FP16|F32|MXFP4(?:_MOE)?))(?:[-_./]|$)`)
)

// ParseQuant extracts the quantization label from a GGUF path ("" if none).
func ParseQuant(p string) string {
	name := shardRe.ReplaceAllString(p, ".gguf")
	name = strings.TrimSuffix(strings.TrimSuffix(name, ".gguf"), ".GGUF")
	ms := quantRe.FindAllStringSubmatch(name, -1)
	if len(ms) == 0 {
		return ""
	}
	q := strings.ToUpper(ms[len(ms)-1][1])
	if q == "FP16" {
		q = "F16"
	}
	return q
}

// IsMMProj reports whether a GGUF file is a vision projector.
func IsMMProj(p string) bool {
	return strings.Contains(strings.ToLower(path.Base(p)), "mmproj")
}

// GGUFVariants groups a repo's GGUF files into variants (shards combined) and returns
// them sorted by size, plus the vision projector files.
func GGUFVariants(files []File) (variants []Variant, mmproj []File) {
	groups := map[string]*Variant{}
	var order []string
	for _, f := range files {
		if !strings.HasSuffix(strings.ToLower(f.Path), ".gguf") {
			continue
		}
		if IsMMProj(f.Path) {
			mmproj = append(mmproj, f)
			continue
		}
		key := shardRe.ReplaceAllString(f.Path, "")
		v, ok := groups[key]
		if !ok {
			v = &Variant{Quant: ParseQuant(f.Path)}
			groups[key] = v
			order = append(order, key)
		}
		v.Files = append(v.Files, f)
		v.Size += f.Size
	}
	for _, k := range order {
		v := groups[k]
		sort.Slice(v.Files, func(i, j int) bool { return v.Files[i].Path < v.Files[j].Path })
		variants = append(variants, *v)
	}
	sort.SliceStable(variants, func(i, j int) bool { return variants[i].Size < variants[j].Size })
	return variants, mmproj
}

// DefaultQuantPreference is the order tried when the user gives no --quant. Q4_K_M is
// first because it is the de-facto standard: results stay comparable across machines
// and downloads stay reasonable. Larger quants are only chosen if Q4-class ones are
// missing; smaller ones only if nothing in this list fits.
var DefaultQuantPreference = []string{
	"Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "Q4_0", "IQ4_XS", "IQ4_NL", "Q4_1",
	"Q5_K_M", "UD-Q5_K_XL", "Q5_K_S", "Q5_0", "Q6_K", "UD-Q6_K_XL", "Q8_0",
	"Q3_K_L", "Q3_K_M", "UD-Q3_K_XL", "IQ3_M", "Q3_K_S", "Q2_K", "IQ2_M",
}

// FindVariant returns the variant with the given quant (case-insensitive).
func FindVariant(vs []Variant, quant string) (Variant, bool) {
	q := strings.ToUpper(quant)
	for _, v := range vs {
		if v.Quant == q {
			return v, true
		}
	}
	return Variant{}, false
}

// RankVariants orders variants by preference: DefaultQuantPreference first, then any
// remaining non-full-precision variants largest first, then full precision.
func RankVariants(vs []Variant) []Variant {
	var out []Variant
	used := map[int]bool{}
	for _, q := range DefaultQuantPreference {
		for i, v := range vs {
			if !used[i] && v.Quant == q {
				out = append(out, v)
				used[i] = true
			}
		}
	}
	var rest, full []Variant
	for i, v := range vs {
		if used[i] {
			continue
		}
		switch v.Quant {
		case "BF16", "F16", "F32":
			full = append(full, v)
		default:
			rest = append(rest, v)
		}
	}
	sort.SliceStable(rest, func(i, j int) bool { return rest[i].Size > rest[j].Size })
	sort.SliceStable(full, func(i, j int) bool { return full[i].Size < full[j].Size })
	return append(append(out, rest...), full...)
}

// PickMMProj prefers an F16 projector, then BF16, then the smallest.
func PickMMProj(files []File) *File {
	if len(files) == 0 {
		return nil
	}
	for _, want := range []string{"F16", "BF16"} {
		for i, f := range files {
			if ParseQuant(f.Path) == want {
				return &files[i]
			}
		}
	}
	best := files[0]
	for _, f := range files[1:] {
		if f.Size < best.Size {
			best = f
		}
	}
	return &best
}
