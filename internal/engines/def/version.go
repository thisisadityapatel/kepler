package def

import (
	"fmt"
	"strconv"
	"strings"
)

// Constraint is a single version comparison such as ">=14.0".
type Constraint struct {
	Op      string // >=, >, <=, <, ==
	Version string
}

// ParseConstraint parses ">=14.0", ">12.4", "==8.9" or a bare "14.0" (meaning >=).
func ParseConstraint(s string) (Constraint, error) {
	s = strings.TrimSpace(s)
	for _, op := range []string{">=", "<=", "==", ">", "<"} {
		if strings.HasPrefix(s, op) {
			v := strings.TrimSpace(s[len(op):])
			if _, err := parseVersion(v); err != nil {
				return Constraint{}, fmt.Errorf("invalid version constraint %q", s)
			}
			return Constraint{Op: op, Version: v}, nil
		}
	}
	if _, err := parseVersion(s); err != nil {
		return Constraint{}, fmt.Errorf("invalid version constraint %q", s)
	}
	return Constraint{Op: ">=", Version: s}, nil
}

// Satisfied reports whether have meets the constraint. Unparseable have → false.
func (c Constraint) Satisfied(have string) bool {
	cmp, err := CompareVersions(have, c.Version)
	if err != nil {
		return false
	}
	switch c.Op {
	case ">=":
		return cmp >= 0
	case ">":
		return cmp > 0
	case "<=":
		return cmp <= 0
	case "<":
		return cmp < 0
	case "==":
		return cmp == 0
	}
	return false
}

// CompareVersions compares dotted numeric versions ("26.6.2" vs "14.0"); missing
// components count as zero. A leading "v" or "b" (llama.cpp build tags) is ignored.
func CompareVersions(a, b string) (int, error) {
	pa, err := parseVersion(a)
	if err != nil {
		return 0, err
	}
	pb, err := parseVersion(b)
	if err != nil {
		return 0, err
	}
	for i := 0; i < len(pa) || i < len(pb); i++ {
		var x, y int
		if i < len(pa) {
			x = pa[i]
		}
		if i < len(pb) {
			y = pb[i]
		}
		if x != y {
			if x < y {
				return -1, nil
			}
			return 1, nil
		}
	}
	return 0, nil
}

func parseVersion(s string) ([]int, error) {
	s = strings.TrimPrefix(strings.TrimPrefix(strings.TrimSpace(s), "v"), "b")
	if s == "" {
		return nil, fmt.Errorf("empty version")
	}
	parts := strings.Split(s, ".")
	out := make([]int, len(parts))
	for i, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid version %q", s)
		}
		out[i] = n
	}
	return out, nil
}
