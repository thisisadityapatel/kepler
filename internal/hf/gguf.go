package hf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

// ArchInfo is what preflight needs to estimate memory.
type ArchInfo struct {
	Architecture string `json:"architecture,omitempty"`
	Layers       int    `json:"layers"`
	KVHeads      int    `json:"kv_heads"`
	HeadDim      int    `json:"head_dim"`
	ContextMax   int    `json:"context_max"`
	Params       int64  `json:"params,omitempty"`
}

// Known reports whether enough is known to estimate the KV cache.
func (a ArchInfo) Known() bool { return a.Layers > 0 && a.KVHeads > 0 && a.HeadDim > 0 }

const ggufMagic = 0x46554747 // "GGUF" little-endian

// GGUF metadata value types.
const (
	gUint8 = iota
	gInt8
	gUint16
	gInt16
	gUint32
	gInt32
	gFloat32
	gBool
	gString
	gArray
	gUint64
	gInt64
	gFloat64
)

// maxHeaderBytes bounds how much of a remote file ReadGGUFArch will read.
const maxHeaderBytes = 64 << 20

// ReadGGUFArch parses the GGUF header from r (the start of the file) and returns
// architecture info. It stops as soon as the needed keys are known, so for remote
// files only the first few KB–MB are transferred.
func ReadGGUFArch(r io.Reader) (ArchInfo, error) {
	br := bufio.NewReaderSize(io.LimitReader(r, maxHeaderBytes), 64<<10)
	var hdr struct {
		Magic   uint32
		Version uint32
	}
	if err := binary.Read(br, binary.LittleEndian, &hdr); err != nil {
		return ArchInfo{}, fmt.Errorf("read GGUF header: %w", err)
	}
	if hdr.Magic != ggufMagic {
		return ArchInfo{}, errors.New("not a GGUF file")
	}
	if hdr.Version < 2 || hdr.Version > 3 {
		return ArchInfo{}, fmt.Errorf("unsupported GGUF version %d", hdr.Version)
	}
	var counts struct{ Tensors, KVs uint64 }
	if err := binary.Read(br, binary.LittleEndian, &counts); err != nil {
		return ArchInfo{}, err
	}

	p := ggufParser{r: br}
	kv := map[string]int64{}
	var arch string
	haveCore := func() bool {
		return arch != "" && kv[arch+".block_count"] > 0 && kv[arch+".attention.head_count"] > 0 &&
			kv[arch+".embedding_length"] > 0 && kv[arch+".context_length"] > 0
	}
	for i := uint64(0); i < counts.KVs; i++ {
		key, err := p.str()
		if err != nil {
			return ArchInfo{}, err
		}
		// Tokenizer arrays are large and come after the model keys: stop early.
		if strings.HasPrefix(key, "tokenizer.") && haveCore() {
			break
		}
		typ, err := p.u32()
		if err != nil {
			return ArchInfo{}, err
		}
		if key == "general.architecture" && typ == gString {
			if arch, err = p.str(); err != nil {
				return ArchInfo{}, err
			}
			continue
		}
		n, ok, err := p.value(typ)
		if err != nil {
			return ArchInfo{}, fmt.Errorf("GGUF key %s: %w", key, err)
		}
		if ok {
			kv[key] = n
		}
	}
	if arch == "" {
		return ArchInfo{}, errors.New("GGUF has no general.architecture")
	}
	a := ArchInfo{
		Architecture: arch,
		Layers:       int(kv[arch+".block_count"]),
		KVHeads:      int(kv[arch+".attention.head_count_kv"]),
		ContextMax:   int(kv[arch+".context_length"]),
		HeadDim:      int(kv[arch+".attention.key_length"]),
		Params:       kv["general.parameter_count"],
	}
	heads, embd := kv[arch+".attention.head_count"], kv[arch+".embedding_length"]
	if a.KVHeads == 0 {
		a.KVHeads = int(heads) // no GQA key: one KV head per attention head
	}
	if a.HeadDim == 0 && heads > 0 {
		a.HeadDim = int(embd / heads)
	}
	return a, nil
}

type ggufParser struct{ r *bufio.Reader }

func (p ggufParser) u32() (uint32, error) {
	var v uint32
	err := binary.Read(p.r, binary.LittleEndian, &v)
	return v, err
}

func (p ggufParser) u64() (uint64, error) {
	var v uint64
	err := binary.Read(p.r, binary.LittleEndian, &v)
	return v, err
}

func (p ggufParser) str() (string, error) {
	n, err := p.u64()
	if err != nil {
		return "", err
	}
	if n > 1<<20 {
		return "", fmt.Errorf("string too long (%d)", n)
	}
	b := make([]byte, n)
	_, err = io.ReadFull(p.r, b)
	return string(b), err
}

func (p ggufParser) skip(n uint64) error {
	_, err := p.r.Discard(int(n))
	return err
}

var scalarSize = map[uint32]uint64{
	gUint8: 1, gInt8: 1, gBool: 1, gUint16: 2, gInt16: 2,
	gUint32: 4, gInt32: 4, gFloat32: 4, gUint64: 8, gInt64: 8, gFloat64: 8,
}

// value reads a value of type typ. Integer scalars are returned with ok=true; integer
// arrays (e.g. per-layer head_count_kv) return their max; everything else is skipped.
func (p ggufParser) value(typ uint32) (int64, bool, error) {
	switch typ {
	case gString:
		_, err := p.str()
		return 0, false, err
	case gArray:
		et, err := p.u32()
		if err != nil {
			return 0, false, err
		}
		n, err := p.u64()
		if err != nil {
			return 0, false, err
		}
		if et == gString {
			for i := uint64(0); i < n; i++ {
				if _, err := p.str(); err != nil {
					return 0, false, err
				}
			}
			return 0, false, nil
		}
		sz, ok := scalarSize[et]
		if !ok {
			return 0, false, fmt.Errorf("unsupported array element type %d", et)
		}
		if et == gFloat32 || et == gFloat64 || et == gBool || n > 4096 {
			return 0, false, p.skip(n * sz)
		}
		var max int64
		for i := uint64(0); i < n; i++ {
			v, _, err := p.scalar(et)
			if err != nil {
				return 0, false, err
			}
			if v > max {
				max = v
			}
		}
		return max, true, nil
	default:
		return p.scalar(typ)
	}
}

func (p ggufParser) scalar(typ uint32) (int64, bool, error) {
	sz, ok := scalarSize[typ]
	if !ok {
		return 0, false, fmt.Errorf("unknown GGUF type %d", typ)
	}
	b := make([]byte, sz)
	if _, err := io.ReadFull(p.r, b); err != nil {
		return 0, false, err
	}
	switch typ {
	case gUint8, gBool:
		return int64(b[0]), typ != gBool, nil
	case gInt8:
		return int64(int8(b[0])), true, nil
	case gUint16:
		return int64(binary.LittleEndian.Uint16(b)), true, nil
	case gInt16:
		return int64(int16(binary.LittleEndian.Uint16(b))), true, nil
	case gUint32:
		return int64(binary.LittleEndian.Uint32(b)), true, nil
	case gInt32:
		return int64(int32(binary.LittleEndian.Uint32(b))), true, nil
	case gUint64:
		v := binary.LittleEndian.Uint64(b)
		if v > math.MaxInt64 {
			return 0, false, nil
		}
		return int64(v), true, nil
	case gInt64:
		return int64(binary.LittleEndian.Uint64(b)), true, nil
	}
	return 0, false, nil // floats are skipped
}

// MLXConfig is the subset of an MLX/transformers config.json needed for ArchInfo.
type MLXConfig struct {
	ModelType             string     `json:"model_type"`
	NumHiddenLayers       int        `json:"num_hidden_layers"`
	NumAttentionHeads     int        `json:"num_attention_heads"`
	NumKeyValueHeads      int        `json:"num_key_value_heads"`
	HiddenSize            int        `json:"hidden_size"`
	HeadDim               int        `json:"head_dim"`
	MaxPositionEmbeddings int        `json:"max_position_embeddings"`
	TextConfig            *MLXConfig `json:"text_config"`
}

// Arch converts a config.json into ArchInfo (multimodal configs use text_config).
func (c MLXConfig) Arch() ArchInfo {
	if c.NumHiddenLayers == 0 && c.TextConfig != nil {
		t := c.TextConfig.Arch()
		if t.Architecture == "" {
			t.Architecture = c.ModelType
		}
		return t
	}
	a := ArchInfo{Architecture: c.ModelType, Layers: c.NumHiddenLayers, KVHeads: c.NumKeyValueHeads,
		HeadDim: c.HeadDim, ContextMax: c.MaxPositionEmbeddings}
	if a.KVHeads == 0 {
		a.KVHeads = c.NumAttentionHeads
	}
	if a.HeadDim == 0 && c.NumAttentionHeads > 0 {
		a.HeadDim = c.HiddenSize / c.NumAttentionHeads
	}
	return a
}
