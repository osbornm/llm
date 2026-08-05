# llm

A large language model written from scratch — including the tokenizer — as a
learning exercise to understand how every piece works.

## Layout

Each piece of the puzzle is its own Go package:

- `tokenizer/` — byte-level BPE tokenizer
- `model/` — transformer architecture (embeddings, attention, feed-forward)
- `train/` — training loop, loss, backpropagation
- `generate/` — inference and sampling
- `cmd/llm/` — CLI entry point (`go run ./cmd/llm`)

## Roadmap

- [ ] Byte-level BPE tokenizer (in progress)
- [ ] Model architecture
- [ ] Training
- [ ] Inference

## Training data

The datasets used to train the tokenizer live in `data/`, which is gitignored.
After cloning the repo, download them:

```sh
mkdir -p data

# Tiny Shakespeare (~1.1 MB) — small dev-loop dataset, trains in seconds
curl -L -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# enwik8 (~95 MB uncompressed) — first 10^8 bytes of English Wikipedia XML,
# messy real-world text for stress-testing
curl -L -o data/enwik8.zip https://mattmahoney.net/dc/enwik8.zip
unzip -o data/enwik8.zip -d data
rm data/enwik8.zip
```

Optional extras:

```sh
# TinyStories validation split (~19 MB) — clean synthetic English prose
curl -L -o data/tinystories-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
```

Use `tinyshakespeare.txt` while developing (fast iteration), then `enwik8` to
train a larger vocabulary and stress-test performance and edge cases
(XML markup, entities, multilingual fragments).
