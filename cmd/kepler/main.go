// Command kepler benchmarks LLMs across every inference engine the machine can run.
package main

import (
	"os"

	"github.com/thisisadityapatel/kepler/internal/cli"
)

func main() { os.Exit(cli.Execute()) }
