package cli

import (
	"testing"

	"github.com/thisisadityapatel/kepler/definitions"
	"github.com/thisisadityapatel/kepler/internal/engines/adapter"
	"github.com/thisisadityapatel/kepler/internal/engines/def"
	"github.com/thisisadityapatel/kepler/internal/engines/hooks/ollama"
	"github.com/thisisadityapatel/kepler/internal/install"
)

// Every shipped definition must validate against the real installer kinds, adapter
// hooks, and canaries wired up by the composition root.
func TestShippedDefinitionsValidateAgainstRegistries(t *testing.T) {
	ollama.Register()
	canaries, err := def.LoadCanaries(definitions.Root(), definitions.CanariesFile)
	if err != nil {
		t.Fatal(err)
	}
	set, err := def.Load(definitions.Engines(), "", def.ValidateOptions{
		InstallKinds:  install.Kinds(),
		Adapters:      adapter.Names(),
		CanaryFormats: canaries.Formats(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(set.Defs) < 3 {
		t.Fatalf("only %d definitions loaded", len(set.Defs))
	}
}
