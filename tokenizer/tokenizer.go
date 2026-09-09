// Package tokenizer implements a byte-level BPE (byte pair encoding)
// tokenizer, trained from scratch on raw text.
package tokenizer

import (
	"errors"
	"slices"
)

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

// Train learns a BPE vocabulary of up to vocabSize from text and returns the
// trained tokenizer. The base vocabulary is the 256 byte values, so vocabSize
// must be at least 256; each merge above that adds one token. Training stops
// early if no adjacent pairs remain.
func Train(text []byte, vocabSize int) (*BPE, error) {
	if vocabSize < 256 {
		return nil, errors.New("tokenizer: vocabSize must be at least 256")
	}
	if vocabSize <= 1<<16 {
		return train[uint16](text, vocabSize), nil
	}
	return train[int](text, vocabSize), nil
}

func train[T uint16 | int](text []byte, vocabSize int) *BPE {
	b := &BPE{Vocab: make([][]byte, 256)}
	for i := range b.Vocab {
		b.Vocab[i] = []byte{byte(i)}
	}
	if vocabSize == 256 || len(text) < 2 {
		return b
	}

	tokens := make([]T, len(text))
	counts := pairCounts{stride: vocabSize}
	// Cap the dense table at 64 MiB, and keep counts within uint32.
	if vocabSize <= 4096 && uint64(len(text)) <= 1<<32 {
		counts.dense = make([]uint32, vocabSize*vocabSize)
	} else {
		counts.sparse = make(map[[2]int]int)
	}
	for i, value := range text {
		tokens[i] = T(value)
		if i > 0 {
			counts.add(int(tokens[i-1]), int(tokens[i]))
		}
	}
	for len(b.Vocab) < vocabSize && len(tokens) > 1 {
		pair := counts.takeBest()
		id := len(b.Vocab)
		b.Vocab = append(b.Vocab, slices.Concat(b.Vocab[pair[0]], b.Vocab[pair[1]]))
		b.Merges = append(b.Merges, pair)
		if len(b.Vocab) == vocabSize {
			break
		}

		left, right, replacement := T(pair[0]), T(pair[1]), T(id)
		merged := tokens[:0]
		for i := 0; i < len(tokens); {
			token := tokens[i]
			if i+1 < len(tokens) && token == left && tokens[i+1] == right {
				token = replacement
				i += 2
			} else {
				i++
			}
			if len(merged) > 0 {
				counts.add(int(merged[len(merged)-1]), int(token))
			}
			merged = append(merged, token)
		}
		tokens = merged
	}

	return b
}

type pairCounts struct {
	stride int
	dense  []uint32
	active []uint32
	sparse map[[2]int]int
}

func (c *pairCounts) add(left, right int) {
	if c.dense != nil {
		key := left*c.stride + right
		if c.dense[key] == 0 {
			c.active = append(c.active, uint32(key))
		}
		c.dense[key]++
	} else {
		c.sparse[[2]int{left, right}]++
	}
}

// takeBest selects the most frequent pair and clears counts for the next pass.
func (c *pairCounts) takeBest() [2]int {
	if c.dense != nil {
		var bestKey, bestCount uint32
		for _, key := range c.active {
			count := c.dense[key]
			if count > bestCount || count == bestCount && key < bestKey {
				bestKey, bestCount = key, count
			}
			c.dense[key] = 0
		}
		c.active = c.active[:0]
		return [2]int{int(bestKey) / c.stride, int(bestKey) % c.stride}
	}

	var pair [2]int
	bestCount := 0
	for candidate, count := range c.sparse {
		if count > bestCount || count == bestCount &&
			(candidate[0] < pair[0] || candidate[0] == pair[0] && candidate[1] < pair[1]) {
			pair, bestCount = candidate, count
		}
	}
	clear(c.sparse)
	return pair
}
