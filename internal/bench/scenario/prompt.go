package scenario

import (
	"fmt"
	"math/rand"
	"strings"
)

// DefaultCharsPerToken is used until the engine's tokenizer has been calibrated.
const DefaultCharsPerToken = 4.0

// corpus is neutral English prose; prompts are built by sampling its sentences with a
// seeded RNG so every engine sees byte-identical text for the same scenario.
var corpus = strings.Fields(`
The harbor town woke slowly under a pale sky.|Fishing boats rocked against the pier while gulls argued over scraps.|
A baker opened her shutters and the smell of bread drifted down the narrow street.|
Children ran past the fountain on their way to school, shouting about a lost kite.|
At the library, an old clock ticked above shelves of maps and travel journals.|
The river that fed the harbor began in mountains three days to the north.|
Merchants once carried salt, wool, and copper along its banks to distant markets.|
Engineers later built locks and bridges, and the town grew around the trade.|
Each spring the council repaired the sea wall after winter storms.|
Farmers in the valley grew barley, apples, and hops for the local brewery.|
A small observatory on the hill offered public evenings when the sky was clear.|
Visitors learned to find the pole star and to tell planets from distant suns.|
The railway arrived late, but it brought teachers, doctors, and new ideas.|
Workshops along the canal made clocks, lenses, and precise measuring tools.|
Every autumn a festival celebrated the harvest with music and long tables of food.|
Historians argue about when the first lighthouse was lit on the rocky point.|
Some records suggest monks kept a fire burning there for sailors long ago.|
Today the lighthouse is automated, and its keeper's cottage is a museum.|
Weather in the region changes quickly, so residents always carry a coat.|
The town's archives hold letters, ledgers, and photographs from many generations.|`)

var sentences = func() []string {
	var out []string
	for _, s := range strings.Split(strings.Join(corpus, " "), "|") {
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}()

// Instruction asks for long output so engines without ignore_eos rarely stop early.
const instruction = "\n\nUsing the text above as background, write a long, detailed, multi-section essay about the town, its history, and its people. Keep writing; do not conclude early."

// Prompt builds the user message for one iteration. The nonce is placed first so no
// prefix is shared between iterations and engine prompt caches cannot help.
func Prompt(s Scenario, nonce string, charsPerToken float64) string {
	if charsPerToken <= 0 {
		charsPerToken = DefaultCharsPerToken
	}
	head := fmt.Sprintf("[session %s]\n", nonce)
	target := int(float64(s.PromptTokens)*charsPerToken) - len(head) - len(instruction)
	var b strings.Builder
	b.WriteString(head)
	rng := rand.New(rand.NewSource(int64(len(s.Name))*7919 + int64(s.PromptTokens)))
	for b.Len()-len(head) < target {
		b.WriteString(sentences[rng.Intn(len(sentences))])
		b.WriteByte(' ')
	}
	b.WriteString(instruction)
	return b.String()
}
