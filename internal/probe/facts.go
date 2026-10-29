// Package probe detects host hardware and produces platform-neutral HostFacts.
// Engine definitions match against HostFacts only, never against OS commands.
package probe

// HostFacts describes the machine. It is serialized into registry.json and every result.
type HostFacts struct {
	OS            string            `json:"os" yaml:"os"`                 // darwin | linux | windows
	OSVersion     string            `json:"os_version" yaml:"os_version"` // "26.0", "22.04"
	Arch          string            `json:"arch" yaml:"arch"`             // arm64 | amd64
	ModelID       string            `json:"model_id,omitempty" yaml:"model_id,omitempty"`
	CPU           CPUInfo           `json:"cpu" yaml:"cpu"`
	RAMBytes      int64             `json:"ram_bytes" yaml:"ram_bytes"`
	Accelerators  []Accelerator     `json:"accelerators" yaml:"accelerators"`
	FreeDiskBytes int64             `json:"free_disk_bytes" yaml:"free_disk_bytes"`
	Power         PowerInfo         `json:"power" yaml:"power"`
	Tools         map[string]string `json:"tools" yaml:"tools"` // name → version ("" if present, version unknown)
}

// CPUInfo describes the CPU.
type CPUInfo struct {
	Brand             string `json:"brand" yaml:"brand"`
	PerformanceCores  int    `json:"performance_cores" yaml:"performance_cores"`
	EfficiencyCores   int    `json:"efficiency_cores" yaml:"efficiency_cores"`
	LogicalProcessors int    `json:"logical_processors" yaml:"logical_processors"`
}

// Accelerator kinds.
const (
	AccelMetal  = "metal"
	AccelCUDA   = "cuda"
	AccelROCm   = "rocm"
	AccelVulkan = "vulkan"
	AccelCPU    = "cpu"
)

// Accelerator is one compute device usable by engines. A cpu accelerator is always present.
type Accelerator struct {
	Kind       string `json:"kind" yaml:"kind"`
	Vendor     string `json:"vendor" yaml:"vendor"`
	Name       string `json:"name" yaml:"name"`
	MemBytes   int64  `json:"mem_bytes" yaml:"mem_bytes"` // usable memory: wired limit on Apple, VRAM elsewhere
	Unified    bool   `json:"unified" yaml:"unified"`
	APIVersion string `json:"api_version,omitempty" yaml:"api_version,omitempty"`
	ComputeCap string `json:"compute_capability,omitempty" yaml:"compute_capability,omitempty"`
	Cores      int    `json:"cores,omitempty" yaml:"cores,omitempty"`
}

// PowerInfo describes laptop power state; both affect benchmark numbers.
type PowerInfo struct {
	Source       string `json:"source" yaml:"source"` // ac | battery | unknown
	LowPowerMode bool   `json:"low_power_mode" yaml:"low_power_mode"`
}

// Accelerator returns the first accelerator of the given kind.
func (h HostFacts) Accelerator(kind string) (Accelerator, bool) {
	for _, a := range h.Accelerators {
		if a.Kind == kind {
			return a, true
		}
	}
	return Accelerator{}, false
}

// PlatformKey is a short display label such as "darwin/arm64/metal".
func (h HostFacts) PlatformKey() string {
	accel := AccelCPU
	for _, a := range h.Accelerators {
		if a.Kind != AccelCPU {
			accel = a.Kind
			break
		}
	}
	return h.OS + "/" + h.Arch + "/" + accel
}
