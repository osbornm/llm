package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// testBPE builds a miniature trained state by hand: the 256 byte tokens plus
// merges producing "th" (id 256) and "the" (id 257).
func testBPE() *BPE {
	b := &BPE{}
	for i := 0; i < 256; i++ {
		b.Vocab = append(b.Vocab, []byte{byte(i)})
	}
	b.Merges = [][2]int{{'t', 'h'}, {256, 'e'}}
	b.Vocab = append(b.Vocab, []byte("th"), []byte("the"))
	return b
}

func TestByteLevelAlphabet(t *testing.T) {
	table := byteLevelAlphabet()
	cases := []struct {
		b    byte
		want rune
	}{
		{'A', 'A'},
		{'!', '!'},
		{'~', '~'},
		{0x00, 0x100}, // first non-printable
		{' ', 'Ġ'},    // space is U+0120 in GPT-2's table
		{'\n', 'Ċ'},
		{0x7F, 0x121}, // DEL, the 34th remapped byte
		{0xAD, 0x143}, // soft hyphen, the last remapped byte
		{0xA1, 0xA1},
		{0xFF, 0xFF},
	}
	for _, c := range cases {
		if got := table[c.b]; got != c.want {
			t.Errorf("table[%#x] = %U, want %U", c.b, got, c.want)
		}
	}
}

func TestSaveHF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := testBPE().SaveHF(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	var got struct {
		Model struct {
			Type   string         `json:"type"`
			Vocab  map[string]int `json:"vocab"`
			Merges []string       `json:"merges"`
		} `json:"model"`
		PreTokenizer struct {
			Type string `json:"type"`
		} `json:"pre_tokenizer"`
		Decoder struct {
			Type string `json:"type"`
		} `json:"decoder"`
	}
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("output is not valid JSON: %v", err)
	}

	if got.Model.Type != "BPE" || got.PreTokenizer.Type != "ByteLevel" || got.Decoder.Type != "ByteLevel" {
		t.Errorf("pipeline types = (%q, %q, %q), want (BPE, ByteLevel, ByteLevel)",
			got.Model.Type, got.PreTokenizer.Type, got.Decoder.Type)
	}
	for text, id := range map[string]int{"t": 't', "Ġ": ' ', "th": 256, "the": 257} {
		if got.Model.Vocab[text] != id {
			t.Errorf("vocab[%q] = %d, want %d", text, got.Model.Vocab[text], id)
		}
	}
	if want := []string{"t h", "th e"}; len(got.Model.Merges) != 2 ||
		got.Model.Merges[0] != want[0] || got.Model.Merges[1] != want[1] {
		t.Errorf("merges = %q, want %q", got.Model.Merges, want)
	}
}

func TestSaveHFRejectsBadState(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.json")

	if err := (&BPE{}).SaveHF(path); err == nil {
		t.Error("empty vocab: want error, got nil")
	}

	b := testBPE()
	b.Merges = append(b.Merges, [2]int{999, 0})
	if err := b.SaveHF(path); err == nil {
		t.Error("out-of-range merge id: want error, got nil")
	}

	b = testBPE()
	b.Vocab = append(b.Vocab, []byte("th"))
	if err := b.SaveHF(path); err == nil {
		t.Error("duplicate token: want error, got nil")
	}

	b = testBPE()
	b.Merges = append(b.Merges, [2]int{'t', 'q'}) // no "tq" vocab entry
	if err := b.SaveHF(path); err == nil {
		t.Error("merge without matching vocab token: want error, got nil")
	}

	b = testBPE()
	b.Vocab[257] = []byte("thx") // merge th+e no longer produces id 257
	if err := b.SaveHF(path); err == nil {
		t.Error("merge/vocab byte mismatch: want error, got nil")
	}
}

func TestSaveHFAtomic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := testBPE().SaveHF(path); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("temp file left behind: %v", entries)
	}
}
