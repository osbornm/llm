// Command llm is the CLI entry point for training the tokenizer and model,
// and for running inference.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/osbornm/llm/tokenizer"
)

func main() {
	dataPath := flag.String("data", "data/tinyshakespeare.txt", "path to training text")
	vocabSize := flag.Int("vocab", 512, "target vocabulary size (min 256)")
	outPath := flag.String("out", "tokenizer.json", "where to write the trained tokenizer (Hugging Face tokenizer.json format)")
	flag.Parse()

	text, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("reading training data: %v\nsee the README for how to download the datasets", err)
	}
	fmt.Printf("training tokenizer on %s (%d bytes), target vocab size %d\n", *dataPath, len(text), *vocabSize)

	tok, err := tokenizer.Train(text, *vocabSize)
	if err != nil {
		log.Fatal(err)
	}

	// Sanity check: a trained tokenizer must round-trip text losslessly.
	sample := "To be, or not to be, that is the question."
	ids := tok.Encode(sample)
	fmt.Printf("sample: %q -> %d tokens\n", sample, len(ids))
	if got := tok.Decode(ids); got != sample {
		log.Fatalf("round-trip failed: got %q", got)
	}
	fmt.Println("round-trip OK")

	if err := tok.SaveHF(*outPath); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("saved %s\n", *outPath)
}
