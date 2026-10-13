//go:build darwin

package probe

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func fakeRunner(outputs map[string]string) Runner {
	return func(_ context.Context, name string, args ...string) (string, error) {
		key := strings.TrimSpace(name + " " + strings.Join(args, " "))
		if out, ok := outputs[key]; ok {
			return out, nil
		}
		return "", fmt.Errorf("no fake for %q", key)
	}
}

func TestProbeDarwinM4(t *testing.T) {
	run := fakeRunner(map[string]string{
		"sysctl -n machdep.cpu.brand_string":  "Apple M4",
		"sysctl -n hw.model":                  "Mac16,1",
		"sysctl -n hw.perflevel0.physicalcpu": "4",
		"sysctl -n hw.perflevel1.physicalcpu": "6",
		"sysctl -n hw.memsize":                "17179869184",
		"sysctl -n iogpu.wired_limit_mb":      "0",
		"sw_vers -productVersion":             "26.6.2",
		"pmset -g batt":                       "Now drawing from 'AC Power'\n -InternalBattery-0 100%",
		"pmset -g":                            "System-wide power settings:\n lowpowermode         0\n",
		"ioreg -rc AGXAccelerator -d1":        `  "gpu-core-count" = 10`,
	})
	h, err := Probe(context.Background(), Options{Run: run})
	if err != nil {
		t.Fatal(err)
	}
	if h.CPU.Brand != "Apple M4" || h.CPU.PerformanceCores != 4 || h.CPU.EfficiencyCores != 6 {
		t.Errorf("cpu = %+v", h.CPU)
	}
	if h.OSVersion != "26.6.2" || h.RAMBytes != 16<<30 || h.Power.Source != "ac" || h.Power.LowPowerMode {
		t.Errorf("facts = %+v", h)
	}
	if h.Arch == "arm64" {
		m, ok := h.Accelerator(AccelMetal)
		if !ok {
			t.Fatal("metal accelerator missing")
		}
		if m.Cores != 10 || m.MemBytes != (16<<30)*2/3 || !m.Unified {
			t.Errorf("metal = %+v", m)
		}
	}
	if _, ok := h.Accelerator(AccelCPU); !ok {
		t.Error("cpu accelerator must always be present")
	}
}

func TestEffectiveWiredLimit(t *testing.T) {
	if got := EffectiveWiredLimit(64<<30, 0); got != 48<<30 {
		t.Errorf("64GiB default = %d", got)
	}
	if got := EffectiveWiredLimit(16<<30, 12288); got != 12<<30 {
		t.Errorf("override = %d", got)
	}
}

func TestParseLowPowerAndBattery(t *testing.T) {
	if !parseLowPowerMode(" lowpowermode         1\n") {
		t.Error("low power mode not detected")
	}
	if got := parsePowerSource("Now drawing from 'Battery Power'"); got != "battery" {
		t.Errorf("source = %q", got)
	}
}
