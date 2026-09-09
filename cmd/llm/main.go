// Command llm trains the tokenizer and reports training performance.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"time"

	"github.com/osbornm/llm/tokenizer"
)

func main() {
	dataPath := flag.String("data", "data/enwik8", "path to training text")
	vocabSize := flag.Int("vocab", 512, "target vocabulary size (min 256)")
	flag.Parse()

	text, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("reading training data: %v\nsee the README for how to download the datasets", err)
	}
	fmt.Printf("training tokenizer on %s (%d bytes), target vocab size %d\n", *dataPath, len(text), *vocabSize)

	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	start := time.Now()
	tok, err := tokenizer.Train(text, *vocabSize)
	elapsed := time.Since(start)
	runtime.ReadMemStats(&after)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("trained tokenizer with %d tokens (%d merges) in %s\n", len(tok.Vocab), len(tok.Merges), elapsed)
	fmt.Printf("throughput: %.2f MB/s; allocated: %.2f MB; allocations: %d\n",
		float64(len(text))/1e6/elapsed.Seconds(),
		float64(after.TotalAlloc-before.TotalAlloc)/1e6,
		after.Mallocs-before.Mallocs)
}
