// Package definitions embeds the shipped engine definitions and canary models.
// Add an engine by adding engines/<name>.yaml; see internal/engines/def for the schema.
package definitions

import (
	"embed"
	"io/fs"
)

//go:embed engines/*.yaml canaries.yaml
var files embed.FS

// Engines is the filesystem of builtin engine definitions (one *.yaml per engine at its root).
func Engines() fs.FS {
	sub, err := fs.Sub(files, "engines")
	if err != nil {
		panic(err) // embedded path is fixed at compile time
	}
	return sub
}

// Root exposes canaries.yaml.
func Root() fs.FS { return files }

// CanariesFile is the name of the canary definition file in Root().
const CanariesFile = "canaries.yaml"
