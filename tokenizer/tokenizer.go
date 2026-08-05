// Package tokenizer implements a byte-level BPE (byte pair encoding)
// tokenizer, trained from scratch on raw text.
package tokenizer

import "errors"

// Tokenizer encodes text to token IDs and decodes token IDs back to text.
type Tokenizer interface {
	Encode(text string) []int
	Decode(ids []int) string
}

// BPE is a byte-level BPE tokenizer: a vocabulary of byte sequences and the
// ordered list of merges that produced it.
type BPE struct {
	// Vocab maps token id -> the raw bytes that token represents.
	// IDs 0-255 are the single-byte base tokens; merge i adds token 256+i.
	Vocab [][]byte
	// Merges lists the learned merges in training order. Each entry is the
	// pair of token ids whose concatenation forms the new token.
	Merges [][2]int
}

var _ Tokenizer = (*BPE)(nil)

func (b *BPE) Encode(text string) []int {
	// TODO: implement BPE encoding.
	panic("tokenizer: Encode not implemented")
}

func (b *BPE) Decode(ids []int) string {
	// TODO: implement BPE decoding.
	panic("tokenizer: Decode not implemented")
}

// Train learns a BPE vocabulary of size vocabSize from text and returns the
// trained tokenizer. The base vocabulary is the 256 byte values, so vocabSize
// must be at least 256; each merge above that adds one token.
func Train(text []byte, vocabSize int) (*BPE, error) {
	if vocabSize < 256 {
		return nil, errors.New("tokenizer: vocabSize must be at least 256")
	}
	// TODO: implement BPE training.
	return nil, errors.New("tokenizer: Train not implemented")
}
