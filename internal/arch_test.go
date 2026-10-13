package internal_test

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

const module = "github.com/thisisadityapatel/kepler/"

// Module boundary (DESIGN.md §1): the shared core never imports the bench module, so a
// future inference module can reuse it unchanged. Only the composition root (cli) may
// import modules.
func TestSharedCoreDoesNotImportModules(t *testing.T) {
	modules := []string{"internal/bench"}
	allowed := map[string]bool{"cli": true, "bench": true}
	err := filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".go") {
			return err
		}
		top := strings.Split(filepath.ToSlash(path), "/")[0]
		if allowed[top] {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		for _, imp := range f.Imports {
			p, _ := strconv.Unquote(imp.Path.Value)
			for _, m := range modules {
				if strings.HasPrefix(p, module+m) {
					t.Errorf("%s imports %s: shared core must not depend on %s", path, p, m)
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
