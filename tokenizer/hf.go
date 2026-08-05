package tokenizer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

// hfTokenizer mirrors the tokenizer.json schema of Hugging Face's
// `tokenizers` library, configured as a byte-level BPE pipeline
// (the GPT-2 family layout).
type hfTokenizer struct {
	Version       string       `json:"version"`
	Truncation    any          `json:"truncation"`
	Padding       any          `json:"padding"`
	AddedTokens   []any        `json:"added_tokens"`
	Normalizer    any          `json:"normalizer"`
	PreTokenizer  *hfByteLevel `json:"pre_tokenizer"`
	PostProcessor *hfByteLevel `json:"post_processor"`
	Decoder       *hfByteLevel `json:"decoder"`
	Model         hfModel      `json:"model"`
}

type hfByteLevel struct {
	Type           string `json:"type"`
	AddPrefixSpace bool   `json:"add_prefix_space"`
	TrimOffsets    bool   `json:"trim_offsets"`
	UseRegex       bool   `json:"use_regex"`
}

type hfModel struct {
	Type                    string         `json:"type"`
	Dropout                 any            `json:"dropout"`
	UnkToken                any            `json:"unk_token"`
	ContinuingSubwordPrefix any            `json:"continuing_subword_prefix"`
	EndOfWordSuffix         any            `json:"end_of_word_suffix"`
	FuseUnk                 bool           `json:"fuse_unk"`
	ByteFallback            bool           `json:"byte_fallback"`
	Vocab                   map[string]int `json:"vocab"`
	Merges                  []string       `json:"merges"`
}

// byteLevelAlphabet returns GPT-2's byte-to-unicode table. Tokens in
// tokenizer.json are stored as printable text, so each of the 256 byte values
// needs a printable character: bytes that are already printable-and-not-space
// map to themselves, and the rest (controls, space, DEL, a few Latin-1 gaps)
// map to U+0100 upward in byte order — space (0x20) becomes 'Ġ' (U+0120).
func byteLevelAlphabet() [256]rune {
	var table [256]rune
	n := 0
	for b := 0; b < 256; b++ {
		switch {
		case b >= '!' && b <= '~', b >= 0xA1 && b <= 0xAC, b >= 0xAE && b <= 0xFF:
			table[b] = rune(b)
		default:
			table[b] = rune(256 + n)
			n++
		}
	}
	return table
}

// tokenText renders raw token bytes as the printable string form used in
// tokenizer.json. The mapping never produces a literal space, which is what
// keeps the space-separated merge entries ("Ġ t") unambiguous.
func tokenText(tok []byte, table *[256]rune) string {
	runes := make([]rune, len(tok))
	for i, b := range tok {
		runes[i] = table[b]
	}
	return string(runes)
}

// SaveHF writes the tokenizer to path as a Hugging Face tokenizer.json,
// loadable with tokenizers.Tokenizer.from_file or, alongside a config,
// transformers.AutoTokenizer.
func (b *BPE) SaveHF(path string) error {
	if len(b.Vocab) < 256 {
		return fmt.Errorf("tokenizer: vocab has %d tokens, need the 256 base bytes", len(b.Vocab))
	}
	table := byteLevelAlphabet()

	vocab := make(map[string]int, len(b.Vocab))
	for id, tok := range b.Vocab {
		text := tokenText(tok, &table)
		if prev, dup := vocab[text]; dup {
			return fmt.Errorf("tokenizer: vocab ids %d and %d are both %q", prev, id, text)
		}
		vocab[text] = id
	}

	merges := make([]string, len(b.Merges))
	for i, m := range b.Merges {
		left, right := m[0], m[1]
		if left < 0 || left >= len(b.Vocab) || right < 0 || right >= len(b.Vocab) {
			return fmt.Errorf("tokenizer: merge %d refers to token id out of range: (%d, %d)", i, left, right)
		}
		// Merge i must produce token 256+i (see BPE.Vocab). HF resolves each
		// merge by looking up the pair's concatenation in the vocab, so a
		// mismatch would either fail to load or — if the text exists at a
		// different id — load fine and silently emit different ids than the
		// Go struct. Catch it at save time instead.
		if id := 256 + i; id >= len(b.Vocab) || !bytes.Equal(b.Vocab[id], slices.Concat(b.Vocab[left], b.Vocab[right])) {
			return fmt.Errorf("tokenizer: merge %d (%d, %d) does not produce vocab token %d", i, left, right, 256+i)
		}
		merges[i] = tokenText(b.Vocab[left], &table) + " " + tokenText(b.Vocab[right], &table)
	}

	byteLevel := func(addPrefixSpace bool) *hfByteLevel {
		return &hfByteLevel{Type: "ByteLevel", AddPrefixSpace: addPrefixSpace, TrimOffsets: true, UseRegex: true}
	}
	out := hfTokenizer{
		Version:       "1.0",
		AddedTokens:   []any{},
		PreTokenizer:  byteLevel(false),
		PostProcessor: byteLevel(false),
		Decoder:       byteLevel(false),
		Model: hfModel{
			Type:   "BPE",
			Vocab:  vocab,
			Merges: merges,
		},
	}

	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return fmt.Errorf("tokenizer: encoding tokenizer.json: %w", err)
	}
	return writeFileAtomic(path, append(data, '\n'))
}

// writeFileAtomic writes via a temp file and rename so an interrupted or
// failed write (crash, full disk) never destroys an existing file at path.
func writeFileAtomic(path string, data []byte) error {
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Chmod(0o644); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}
