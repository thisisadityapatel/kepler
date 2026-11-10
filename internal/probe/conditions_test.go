package probe

import "testing"

func TestParseThermal(t *testing.T) {
	cases := map[string]string{
		"Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded": ThermalNominal,
		"CPU_Scheduler_Limit = 100\nCPU_Available_CPUs = 10\nCPU_Speed_Limit = 100":                              ThermalNominal,
		"CPU_Speed_Limit = 72":            ThermalElevated,
		"Thermal warning level set to 2.": ThermalElevated,
		"something unexpected":            ThermalUnknown,
	}
	for in, want := range cases {
		if got, _ := ParseThermal(in); got != want {
			t.Errorf("ParseThermal(%q) = %s, want %s", in, got, want)
		}
	}
}
