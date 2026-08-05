// Package tokenizer implements a byte-level BPE (byte pair encoding)
// tokenizer, trained from scratch on raw text.
package tokenizer

// Tokenizer encodes text to token IDs and decodes token IDs back to text.
type Tokenizer interface {
	Encode(text string) []int
	Decode(ids []int) string
}
