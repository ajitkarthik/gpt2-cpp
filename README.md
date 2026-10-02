# gpt2-cpp

A GPT-2 (124M) inference engine written from scratch in C++23, with no
dependencies beyond the standard library. No BLAS, no Eigen, no ONNX — the
tensor library, the transformer blocks, and the weight loader are all hand
written.

The forward pass is validated layer by layer against Karpathy's
[llm.c](https://github.com/karpathy/llm.c), which is in turn validated against
PyTorch. Every intermediate activation — the token embeddings, the output of
each of the 12 transformer blocks, the final layernorm, and the vocabulary
projection — matches the reference to a relative tolerance of 1e-5.

```
Loaded checkpoint file. Size:474MB
Context length:         1024
Vocabulary size:        50257
Layers:                 12
Attention heads:        12
Embedding dimensions:   768
Padded vocabulary size: 50304
Checking encodings ... PASS
Checking activations for layer 1 ... PASS
...
Checking activations for layer 12 ... PASS
Checking activations for final layernorm ... PASS
Checking activations for final vocab projection ... PASS
```

## Design

**Weights are never copied.** The 475MB checkpoint is `mmap`ed once and every
weight tensor is a non-owning view into that mapping. Pages fault in on demand
and stay in the page cache, so resident memory is only what the forward pass
actually touches.

This drives the central type split:

- `MatrixView` — a non-owning 2D descriptor (pointer plus `rows`, `cols`,
  `row_stride`, `col_stride`). Used for weights, which live in the mapping, and
  for zero-copy slicing of activations.
- `Tensor` — owns its buffer. Used for activations, which are computed rather
  than loaded.

Everything is deliberately 2D. Multi-head attention wants a 4D
`(batch, heads, seq, head_dim)` layout; instead each head is carved out of the
`(T, 3C)` QKV activation as a strided `MatrixView` and treated as a 2D matmul.
The index arithmetic is written by hand on purpose.

Inference only — no autograd, no gradient buffers, no computation graph.

### Layering

```
MappedFile    mmap + RAII. Knows nothing about any file format.
     |
Checkpoint    Owns the mapping. Parses the header, derives the 16 tensor
     |        offsets from the model dimensions, hands out typed spans.
     |        The only code that knows a byte offset exists.
     |
Decoder       Built from a Checkpoint. Owns 12 Layers, the final layernorm,
              and the tied vocabulary projection. Knows the math, not the file.
```

A `Layer` composes `LayerNorm`, `MHSA`, and `FFN`.

`Decoder::forward` takes an optional callback that is invoked with each
intermediate activation. When it is absent the forward pass is unchanged; when
it is supplied, the comparison harness uses it to check against reference data.
File I/O lives in the apps, never in the library.

## Layout

```
include/        tensor.hpp  mappedfile.hpp  checkpoint.hpp  decoder.hpp  reference.hpp
src/            implementations
apps/           generate.cpp          text generation (in progress)
                testactivations.cpp   layer-by-layer validation
tests/          any tests/*.cpp becomes a ctest case
write_tokenizer.py   regenerates gpt2_tokenizer.bin
```

## Building

Requires CMake 3.20+ and a C++23 compiler.

```bash
cmake -S . -B build
cmake --build build
```

Debug is the default so the shape and bounds assertions stay live — they are
what catch a transposed weight matrix or a bad stride. For timed runs:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
```

## Data files

`gpt2_124M.bin` (475MB) is not committed. It is the GPT-2 small checkpoint in
llm.c's format: a 256-int32 header followed by 124,475,904 fp32 parameters.
Generate it by running llm.c's `train_gpt2.py`, which exports it from the
HuggingFace checkpoint.

Committed, because they are small:

| file | size | contents |
|---|---|---|
| `gpt2_tokenizer.bin` | 363KB | 50,257 token strings, length-prefixed. Decode only. |
| `tokens.bin` | 512B | 64 input token ids, the validation input |
| `reference_activations.bin` | 2.9MB | llm.c's activations for those tokens |

`write_tokenizer.py` rebuilds `gpt2_tokenizer.bin` from `tiktoken` if needed.

Encoding (text to token ids) is deliberately not implemented — byte-level BPE
encode has subtleties that belong in their own project. Token ids come in
pre-tokenized; the tokenizer file is used only to turn generated ids back into
text.

## Running the validation

```bash
cd build && ./testactivations
```

It loads the checkpoint, runs one forward pass over the 64 tokens in
`tokens.bin`, and compares all 15 intermediate tensors against
`reference_activations.bin`.

Two details matter when reading the comparison. The logits are 50,304 wide
rather than 50,257 — llm.c pads the vocabulary dimension for tile alignment, so
the trailing 47 entries have no counterpart. And because only the last
position's logits are needed for next-token prediction, the final layernorm and
projection are computed on a single row, which is compared against row 63 of
the reference.

## Status

- [x] `mmap`-backed weight loader with derived offsets and shape validation
- [x] 2D tensor library with strided views
- [x] LayerNorm, GeLU, Linear, causal multi-head self-attention, FFN
- [x] Full 12-layer forward pass, validated against llm.c to 1e-5
- [x] Tokenizer decode table
- [ ] Autoregressive generation loop
- [ ] Sampling: greedy, temperature, top-k
- [ ] KV cache

## License

MIT. See `LICENSE`.
