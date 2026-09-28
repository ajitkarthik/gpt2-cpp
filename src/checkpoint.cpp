#include "checkpoint.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>

using namespace std;

Checkpoint::Checkpoint(const string file) : mp(MappedFile(file.c_str())) {
    /* Format of the checkpoint file:
    Each field below is 4 bytes.
    Little endian format.
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
    constexpr auto OFFSET_MAGIC = 0;                      // Magic number
    constexpr auto OFFSET_VERSION = 1 * sizeof(int32_t);  // Version
    constexpr auto OFFSET_MAXT = 2 * sizeof(int32_t);     // Max context length
    constexpr auto OFFSET_V = 3 * sizeof(int32_t);        // Vocabulary
    constexpr auto OFFSET_L = 4 * sizeof(int32_t);        // # of layers
    constexpr auto OFFSET_NH = 5 * sizeof(int32_t);       // # of heads
    constexpr auto OFFSET_C = 6 * sizeof(int32_t);        // Embedding size
    constexpr auto OFFSET_VP = 7 * sizeof(int32_t);       // Padded vocab length

    constexpr auto MAGIC = 20240326;
    constexpr auto VERSION = 3;

    // Read weights from file
    MappedFile mp_ = MappedFile(file.c_str());
    assert(mp_.to_int32(mp_.bytes().subspan(OFFSET_MAGIC, sizeof(int32_t))) == MAGIC);
    assert(mp_.to_int32(mp_.bytes().subspan(OFFSET_VERSION, sizeof(int32_t))) == VERSION);
    maxT = mp_.to_int32(mp_.bytes().subspan(OFFSET_MAXT, sizeof(int32_t)));
    vocab = mp_.to_int32(mp_.bytes().subspan(OFFSET_V, sizeof(int32_t)));
    layers = mp_.to_int32(mp_.bytes().subspan(OFFSET_L, sizeof(int32_t)));
    nh = mp_.to_int32(mp_.bytes().subspan(OFFSET_NH, sizeof(int32_t)));
    embsz = mp_.to_int32(mp_.bytes().subspan(OFFSET_C, sizeof(int32_t)));
    vocab_padded = mp_.to_int32(mp_.bytes().subspan(OFFSET_VP, sizeof(int32_t)));

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

    offset_wte = 1024;
    offset_wpe = offset_wte + (vocab_padded * embsz) * sizeof(float);
    offset_ln1w = offset_wpe + (maxT * embsz) * sizeof(float);
    offset_ln1b = offset_ln1w + (layers * embsz) * sizeof(float);
    offset_qkvw = offset_ln1b + (layers * embsz) * sizeof(float);
    offset_qkvb = offset_qkvw + (layers * 3 * embsz * embsz) * sizeof(float);
    offset_attprojw = offset_qkvb + (layers * 3 * embsz) * sizeof(float);
    offset_attprojb = offset_attprojw + (layers * embsz * embsz) * sizeof(float);
    offset_ln2w = offset_attprojb + (layers * embsz) * sizeof(float);
    offset_ln2b = offset_ln2w + (layers * embsz) * sizeof(float);
    offset_fcw = offset_ln2b + (layers * embsz) * sizeof(float);
    offset_fcb = offset_fcw + (layers * 4 * embsz * embsz) * sizeof(float);
    offset_fcprojw = offset_fcb + (layers * 4 * embsz) * sizeof(float);
    offset_fcprojb = offset_fcprojw + (layers * embsz * 4 * embsz) * sizeof(float);
    offset_lnfw = offset_fcprojb + (layers * embsz) * sizeof(float);
    offset_lnfb = offset_lnfw + (embsz) * sizeof(float);

    size_wte = offset_wpe - offset_wte;
    size_wpe = offset_ln1w - offset_wpe;
    size_ln1w = offset_ln1b - offset_ln1w;
    size_ln1b = offset_qkvw - offset_ln1b;
    size_qkvw = offset_qkvb - offset_qkvw;
    size_qkvb = offset_attprojw - offset_qkvb;
    size_attprojw = offset_attprojb - offset_attprojw;
    size_attprojb = offset_ln2w - offset_attprojb;
    size_ln2w = offset_ln2b - offset_ln2w;
    size_ln2b = offset_fcw - offset_ln2b;
    size_fcw = offset_fcb - offset_fcw;
    size_fcb = offset_fcprojw - offset_fcb;
    size_fcprojw = offset_fcprojb - offset_fcprojw;
    size_fcprojb = offset_lnfw - offset_fcprojb;
    size_lnfw = offset_lnfb - offset_lnfw;
    size_lnfb = embsz * sizeof(float);

    // Some sanity asserts since weights are known in advance
    // Check if offset math above is correct
    assert(offset_lnfb + (embsz * sizeof(float)) == mp_.bytes().size());
    // Check that token embedding weights are 0 between vocab and vocab_padded
    assert(std::all_of(mp_.bytes().begin() + offset_wte + vocab * embsz * sizeof(float),
                       mp_.bytes().begin() + offset_wte + vocab_padded * embsz * sizeof(float),
                       [](std::byte b) { return b == std::byte{0}; }));
}