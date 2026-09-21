#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>

#include "decoder.hpp"
#include "mappedfile.hpp"
#include "tensor.hpp"

using namespace std;
using tn::MatrixView;
using tn::Tensor;

// Looks up tokens in an embedding matrix -> (T, C)
// Adds positional embeddings -> (T, C)
// Returns a tensor of shape (T, C)
Tensor embed(span<const int32_t> tokens, MatrixView& wte, MatrixView& wpe) {
    int T = static_cast<int>(tokens.size());
    assert(T >= 0 && T <= wpe.rows);
    Tensor x(T, wte.cols);
    for (int i = 0; i < T; i++) {
        assert(tokens[i] >= 0 && tokens[i] < wte.rows);
        for (int j = 0; j < wte.cols; j++) {
            x.set(i, j, wte.at(tokens[i], j) + wpe.at(i, j));
        }
    }
    return x;
}

int main(void) {
    /* Each field below is 4 bytes
    |   idx |    value | meaning                    |
    |------:|---------:|----------------------------|
    |     0 | 20240326 | magic                      |
    |     1 |        3 | version (3 = padded vocab) |
    |     2 |     1024 | `maxT`                     |
    |     3 |    50257 | `V`                        |
    |     4 |       12 | `L`                        |
    |     5 |       12 | `NH`                       |
    |     6 |      768 | `C`                        |
    |     7 |    50304 | `Vp`                       |
    | 8–255 |        0 | padding                    |
    */
    constexpr auto CHECKPOINTFILE = "gpt2_124M.bin";
    constexpr auto OFFSET_MAGIC = 0;                      // Magic number
    constexpr auto OFFSET_VERSION = 1 * sizeof(int32_t);  // Version
    constexpr auto OFFSET_MAXT = 2 * sizeof(int32_t);     // Max context length
    constexpr auto OFFSET_V = 3 * sizeof(int32_t);        // Vocabulary
    constexpr auto OFFSET_L = 4 * sizeof(int32_t);        // # of layers
    constexpr auto OFFSET_NH = 5 * sizeof(int32_t);       // # of heads
    constexpr auto OFFSET_C = 6 * sizeof(int32_t);        // Embedding size
    constexpr auto OFFSET_VP = 7 * sizeof(int32_t);       // Padded vocab length
    constexpr auto OFFSET_PAYLOAD = 1024;                 // Weights start from byte offset 1024

    constexpr auto MAGIC = 20240326;
    constexpr auto VERSION = 3;

    // Read weights from file
    MappedFile mp = MappedFile(CHECKPOINTFILE);
    assert(mp.to_int32(mp.bytes().subspan(OFFSET_MAGIC, sizeof(int32_t))) == MAGIC);
    assert(mp.to_int32(mp.bytes().subspan(OFFSET_VERSION, sizeof(int32_t))) == VERSION);
    int32_t maxT = mp.to_int32(mp.bytes().subspan(OFFSET_MAXT, sizeof(int32_t)));
    int32_t vocab = mp.to_int32(mp.bytes().subspan(OFFSET_V, sizeof(int32_t)));
    int32_t layers = mp.to_int32(mp.bytes().subspan(OFFSET_L, sizeof(int32_t)));
    int32_t nh = mp.to_int32(mp.bytes().subspan(OFFSET_NH, sizeof(int32_t)));
    int32_t embsz = mp.to_int32(mp.bytes().subspan(OFFSET_C, sizeof(int32_t)));
    int32_t vocab_padded = mp.to_int32(mp.bytes().subspan(OFFSET_VP, sizeof(int32_t)));

    cout << "Loaded checkpoint file\n";
    cout << "Context length:         " << maxT << "\n";
    cout << "Vocabulary size:        " << vocab << "\n";
    cout << "Layers:                 " << layers << "\n";
    cout << "Attention heads:        " << nh << "\n";
    cout << "Embedding dimensions:   " << embsz << "\n";
    cout << "Padded vocabulary size: " << vocab_padded << "\n";

    /* fp32 weights ...
    |  # | tensor     | shape      |      count | byte offset |
    |---:|------------|------------|-----------:|------------:|
    |  0 | `wte`      | (Vp, C)    | 38,633,472 |       1,024 |
    |  1 | `wpe`      | (maxT, C)  |    786,432 | 154,534,912 |
    |  2 | `ln1w`     | (L, C)     |      9,216 | 157,680,640 |
    |  3 | `ln1b`     | (L, C)     |      9,216 | 157,717,504 |
    |  4 | `qkvw`     | (L, 3C, C) | 21,233,664 | 157,754,368 |
    |  5 | `qkvb`     | (L, 3C)    |     27,648 | 242,689,024 |
    |  6 | `attprojw` | (L, C, C)  |  7,077,888 | 242,799,616 |
    |  7 | `attprojb` | (L, C)     |      9,216 | 271,111,168 |
    |  8 | `ln2w`     | (L, C)     |      9,216 | 271,148,032 |
    |  9 | `ln2b`     | (L, C)     |      9,216 | 271,184,896 |
    | 10 | `fcw`      | (L, 4C, C) | 28,311,552 | 271,221,760 |
    | 11 | `fcb`      | (L, 4C)    |     36,864 | 384,467,968 |
    | 12 | `fcprojw`  | (L, C, 4C) | 28,311,552 | 384,615,424 |
    | 13 | `fcprojb`  | (L, C)     |      9,216 | 497,861,632 |
    | 14 | `lnfw`     | (C,)       |        768 | 497,898,496 |
    | 15 | `lnfb`     | (C,)       |        768 | 497,901,568 |
    */

    constexpr auto OFFSET_WTE = 1024;
    const auto OFFSET_WPE = OFFSET_WTE + (vocab_padded * embsz) * sizeof(float);
    const auto OFFSET_LN1W = OFFSET_WPE + (maxT * embsz) * sizeof(float);
    const auto OFFSET_LN1B = OFFSET_LN1W + (layers * embsz) * sizeof(float);
    const auto OFFSET_QKVW = OFFSET_LN1B + (layers * embsz) * sizeof(float);
    const auto OFFSET_QKVB = OFFSET_QKVW + (layers * 3 * embsz * embsz) * sizeof(float);
    const auto OFFSET_ATTPROJW = OFFSET_QKVB + (layers * 3 * embsz) * sizeof(float);
    const auto OFFSET_ATTPROJB = OFFSET_ATTPROJW + (layers * embsz * embsz) * sizeof(float);
    const auto OFFSET_LN2W = OFFSET_ATTPROJB + (layers * embsz) * sizeof(float);
    const auto OFFSET_LN2B = OFFSET_LN2W + (layers * embsz) * sizeof(float);
    const auto OFFSET_FCW = OFFSET_LN2B + (layers * embsz) * sizeof(float);
    const auto OFFSET_FCB = OFFSET_FCW + (layers * 4 * embsz * embsz) * sizeof(float);
    const auto OFFSET_FCPROJW = OFFSET_FCB + (layers * 4 * embsz) * sizeof(float);
    const auto OFFSET_FCPROJB = OFFSET_FCPROJW + (layers * embsz * 4 * embsz) * sizeof(float);
    const auto OFFSET_LNFW = OFFSET_FCPROJB + (layers * embsz) * sizeof(float);
    const auto OFFSET_LNFB = OFFSET_LNFW + (embsz) * sizeof(float);

    const auto SIZE_LN1W = OFFSET_LN1B - OFFSET_LN1W;
    const auto SIZE_LN1B = OFFSET_QKVW - OFFSET_LN1B;
    const auto SIZE_QKVW = OFFSET_QKVB - OFFSET_QKVW;
    const auto SIZE_QKVB = OFFSET_ATTPROJW - OFFSET_QKVB;
    const auto SIZE_ATTPROJW = OFFSET_ATTPROJB - OFFSET_ATTPROJW;
    const auto SIZE_ATTPROJB = OFFSET_LN2W - OFFSET_ATTPROJB;
    const auto SIZE_LN2W = OFFSET_LN2B - OFFSET_LN2W;
    const auto SIZE_LN2B = OFFSET_FCW - OFFSET_LN2B;
    const auto SIZE_FCW = OFFSET_FCB - OFFSET_FCW;
    const auto SIZE_FCB = OFFSET_FCPROJW - OFFSET_FCB;
    const auto SIZE_FCPROJW = OFFSET_FCPROJB - OFFSET_FCPROJW;
    const auto SIZE_FCPROJB = OFFSET_LNFW - OFFSET_FCPROJB;
    const auto SIZE_LNFW = OFFSET_LNFB - OFFSET_LNFW;
    const auto SIZE_LNFB = embsz * sizeof(float);

    // Some sanity asserts since weights are known in advance
    // Check if offset math above is correct
    assert(OFFSET_LNFB + (embsz * sizeof(float)) == mp.bytes().size());
    // Check that token embedding weights are 0 between vocab and vocab_padded
    assert(std::all_of(mp.bytes().begin() + OFFSET_WTE + vocab * embsz * sizeof(float),
                       mp.bytes().begin() + OFFSET_WTE + vocab_padded * embsz * sizeof(float),
                       [](std::byte b) { return b == std::byte{0}; }));

    // Instantiate the decoder
    decoder::Decoder GPT2(maxT, vocab, layers, nh, embsz, vocab_padded);

    // Load weights into decoder
    for (int i = 0; i < layers; i++) {
        GPT2.loadLayerWeights(
            i, mp.floats_at(OFFSET_LN1W + (SIZE_LN1W / layers) * i, embsz),
            mp.floats_at(OFFSET_LN1B + (SIZE_LN1B / layers) * i, embsz),
            mp.floats_at(OFFSET_QKVW + (SIZE_QKVW / layers) * i, 3 * embsz * embsz),
            mp.floats_at(OFFSET_QKVB + (SIZE_QKVB / layers) * i, 3 * embsz),
            mp.floats_at(OFFSET_ATTPROJW + (SIZE_ATTPROJW / layers) * i, embsz * embsz),
            mp.floats_at(OFFSET_ATTPROJB + (SIZE_ATTPROJB / layers) * i, embsz),
            mp.floats_at(OFFSET_LN2W + (SIZE_LN2W / layers) * i, embsz),
            mp.floats_at(OFFSET_LN2B + (SIZE_LN2B / layers) * i, embsz),
            mp.floats_at(OFFSET_FCW + (SIZE_FCW / layers) * i, 4 * embsz * embsz),
            mp.floats_at(OFFSET_FCB + (SIZE_FCB / layers) * i, 4 * embsz),
            mp.floats_at(OFFSET_FCPROJW + (SIZE_FCPROJW / layers) * i, embsz * 4 * embsz),
            mp.floats_at(OFFSET_FCPROJB + (SIZE_FCPROJB / layers) * i, embsz));
    }

    GPT2.loadLNWeights(mp.floats_at(OFFSET_LNFW + (SIZE_LNFW / layers), embsz),
                       mp.floats_at(OFFSET_LNFB + (SIZE_LNFB / layers), embsz));

    // wte (vocab_padded, embsz)
    MatrixView wte =
        MatrixView(reinterpret_cast<const float*>(mp.bytes().subspan(OFFSET_PAYLOAD).data()),
                   vocab_padded, embsz);

    // wpe (maxT, embsz)
    MatrixView wpe = MatrixView(
        reinterpret_cast<const float*>(mp.bytes().subspan(OFFSET_WPE).data()), maxT, embsz);

    // Autoregressive loop
    // while (token!= end token) {
    // Lookup the tokens in wte to get a vector (T, embsz)
    // TODO: Needs to be hooked up to the tokenizer
    // span<const int32_t> tokens;
    // Tensor x = embed(tokens, wte, wpe);
}