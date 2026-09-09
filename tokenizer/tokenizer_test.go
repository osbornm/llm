package tokenizer

import (
	"bytes"
	"cmp"
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

func TestTrain(t *testing.T) {
	tests := []struct {
		name      string
		text      string
		vocabSize int
		merges    [][2]int
		tokens    []string
	}{
		{
			name:      "stops at target after repeated merges",
			text:      "ababab",
			vocabSize: 258,
			merges:    [][2]int{{'a', 'b'}, {256, 256}},
			tokens:    []string{"ab", "abab"},
		},
		{
			name:      "replaces overlapping pairs left to right",
			text:      "aaaaa",
			vocabSize: 258,
			merges:    [][2]int{{'a', 'a'}, {256, 'a'}},
			tokens:    []string{"aa", "aaa"},
		},
		{
			name:      "counts both boundaries of merged tokens",
			text:      "xababx",
			vocabSize: 259,
			merges:    [][2]int{{'a', 'b'}, {'x', 256}, {256, 'x'}},
			tokens:    []string{"ab", "xab", "abx"},
		},
		{
			name:      "keeps unaffected pairs eligible after merging",
			text:      "ababcd",
			vocabSize: 258,
			merges:    [][2]int{{'a', 'b'}, {'c', 'd'}},
			tokens:    []string{"ab", "cd"},
		},
		{
			name:      "breaks frequency ties by left token",
			text:      "bac",
			vocabSize: 257,
			merges:    [][2]int{{'a', 'c'}},
			tokens:    []string{"ac"},
		},
		{
			name:      "breaks frequency ties by right token",
			text:      "acab",
			vocabSize: 257,
			merges:    [][2]int{{'a', 'b'}},
			tokens:    []string{"ab"},
		},
		{
			name:      "prefers frequency over smaller token IDs",
			text:      "abcbcbc",
			vocabSize: 257,
			merges:    [][2]int{{'b', 'c'}},
			tokens:    []string{"bc"},
		},
		{
			name:      "stops when no pairs remain",
			text:      "ababab",
			vocabSize: 260,
			merges:    [][2]int{{'a', 'b'}, {256, 256}, {257, 256}},
			tokens:    []string{"ab", "abab", "ababab"},
		},
		{name: "empty input", vocabSize: 258},
		{name: "single byte", text: "a", vocabSize: 258},
		{name: "base vocabulary only", text: "ababab", vocabSize: 256},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := []byte(tt.text)
			original := bytes.Clone(input)
			bpe, err := Train(input, tt.vocabSize)
			if err != nil {
				t.Fatal(err)
			}
			if bpe == nil {
				t.Fatal("Train returned a nil tokenizer")
			}
			if !bytes.Equal(input, original) {
				t.Fatalf("Train modified input: got %q, want %q", input, original)
			}
			if !slices.Equal(bpe.Merges, tt.merges) {
				t.Errorf("merges = %v, want %v", bpe.Merges, tt.merges)
			}
			if len(bpe.Vocab) != 256+len(tt.tokens) {
				t.Fatalf("vocabulary size = %d, want %d", len(bpe.Vocab), 256+len(tt.tokens))
			}
			for id := 0; id < 256; id++ {
				if !bytes.Equal(bpe.Vocab[id], []byte{byte(id)}) {
					t.Fatalf("base token %d = %v, want [%d]", id, bpe.Vocab[id], id)
				}
			}
			for i, want := range tt.tokens {
				if got := string(bpe.Vocab[256+i]); got != want {
					t.Errorf("token %d = %q, want %q", 256+i, got, want)
				}
			}
		})
	}
}

func TestTrainRejectsSmallVocabulary(t *testing.T) {
	bpe, err := Train([]byte("abab"), 255)
	if err == nil || bpe != nil {
		t.Fatalf("Train with vocabulary size 255 = (%v, %v), want (nil, error)", bpe, err)
	}
}

func TestTrainMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	corpora := []struct {
		name string
		text []byte
	}{
		{name: "small alphabet", text: make([]byte, 512)},
		{name: "all byte values", text: make([]byte, 768)},
		{name: "frequency ties", text: make([]byte, 256)},
	}
	alphabet := []byte{0, 1, 2, 'a', 'b', 'c', 254, 255}
	for i := range corpora[0].text {
		corpora[0].text[i] = alphabet[rng.Intn(len(alphabet))]
	}
	for i := range corpora[1].text {
		corpora[1].text[i] = byte(rng.Intn(256))
	}
	for i := range corpora[2].text {
		corpora[2].text[i] = byte(i)
	}

	for _, corpus := range corpora {
		for _, vocabSize := range []int{257, 272, 300} {
			t.Run(fmt.Sprintf("%s/%d", corpus.name, vocabSize), func(t *testing.T) {
				assertTrainMatchesReference(t, corpus.text, vocabSize)
			})
		}
	}
}

func TestTrainLargeRequestedVocabulary(t *testing.T) {
	text := []byte{0, 255, 0, 255, 0, 0, 255, 255, 0, 255}
	for _, vocabSize := range []int{4097, 65536, 65537} {
		t.Run(fmt.Sprint(vocabSize), func(t *testing.T) {
			assertTrainMatchesReference(t, text, vocabSize)
		})
	}
}

func assertTrainMatchesReference(t *testing.T, text []byte, vocabSize int) {
	t.Helper()
	original := bytes.Clone(text)
	want := referenceTrain(text, vocabSize)
	got, err := Train(text, vocabSize)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(text, original) {
		t.Fatal("Train modified its input")
	}
	if !slices.Equal(got.Merges, want.Merges) {
		t.Fatalf("merges = %v, want %v", got.Merges, want.Merges)
	}
	if !slices.EqualFunc(got.Vocab, want.Vocab, bytes.Equal) {
		t.Fatalf("vocabulary = %v, want %v", got.Vocab, want.Vocab)
	}
}

// referenceTrain recounts and sorts all pairs each round and writes replacements
// into a fresh slice, independently of Train's optimized storage and counting.
func referenceTrain(text []byte, vocabSize int) *BPE {
	b := &BPE{Vocab: make([][]byte, 256)}
	for id := range b.Vocab {
		b.Vocab[id] = []byte{byte(id)}
	}
	tokens := make([]int, len(text))
	for i, value := range text {
		tokens[i] = int(value)
	}
	for len(b.Vocab) < vocabSize && len(tokens) > 1 {
		counts := make(map[[2]int]int)
		for i := 1; i < len(tokens); i++ {
			counts[[2]int{tokens[i-1], tokens[i]}]++
		}
		pairs := make([][2]int, 0, len(counts))
		for pair := range counts {
			pairs = append(pairs, pair)
		}
		slices.SortFunc(pairs, func(a, b [2]int) int {
			if frequency := cmp.Compare(counts[b], counts[a]); frequency != 0 {
				return frequency
			}
			if left := cmp.Compare(a[0], b[0]); left != 0 {
				return left
			}
			return cmp.Compare(a[1], b[1])
		})
		pair := pairs[0]
		id := len(b.Vocab)
		b.Vocab = append(b.Vocab, slices.Concat(b.Vocab[pair[0]], b.Vocab[pair[1]]))
		b.Merges = append(b.Merges, pair)
		merged := make([]int, 0, len(tokens))
		for i := 0; i < len(tokens); i++ {
			if i+1 < len(tokens) && tokens[i] == pair[0] && tokens[i+1] == pair[1] {
				merged = append(merged, id)
				i++
			} else {
				merged = append(merged, tokens[i])
			}
		}
		tokens = merged
	}
	return b
}
