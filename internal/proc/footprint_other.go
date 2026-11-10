//go:build !darwin

package proc

import "errors"

var errNoFootprint = errors.New("memory footprint sampling is not implemented on this OS yet")

// Footprint is not implemented outside macOS yet.
func Footprint(pid int) (int64, error) { return 0, errNoFootprint }

// GroupPIDs is not implemented outside macOS yet.
func GroupPIDs(pgid int) ([]int, error) { return nil, errNoFootprint }

// GroupFootprint is not implemented outside macOS yet.
func GroupFootprint(pgid int) (int64, error) { return 0, errNoFootprint }

// GroupMappedResident is not implemented outside macOS yet.
func GroupMappedResident(pgid int, prefix string) (int64, error) { return 0, errNoFootprint }

// MappedResident is not implemented outside macOS yet.
func MappedResident(pid int, prefix string) (int64, error) { return 0, errNoFootprint }

// FootprintMethod names the metric for result files.
const FootprintMethod = "unavailable"
