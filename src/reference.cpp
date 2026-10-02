// ** reference_activations.bin FILE FORMAT (little endian, fp32) **
//
// Reference activations from Karpathy's llm.c (test_gpt2.c), step-0 forward pass
// over the 64 token ids in tokens.bin. Batch 0 only. llm.c's own logits, loss and
// gradient checks against the PyTorch reference passed on the producing run.
//
// header: bytes 0-1023 (256 x int32)
//   [0] magic       = 20260928
//   [1] version     = 1
//   [2] B           = 1
//   [3] T           = 64
//   [4] C           = 768
//   [5] L           = 12
//   [6] Vp          = 50304   (PADDED vocab -- logits are this wide, not 50257)
//   [7] num_tensors = 15
//   [8] logit_rows  = 1       (tensor 14 holds ONLY position T-1)
//   [9..255] zero
//
// payload: bytes 1024 onwards. 15 tensors back to back, no per-tensor headers.
// Tensors 0..13 are row-major fp32, shape (T, C): element (t,c) at t*C + c.
// Tensor 14 is a single row of Vp floats -- the logits at position T-1 only.
//
//   idx  tensor                             shape       floats    byte offset
//   ---  ---------------------------------- ----------  --------  -----------
//     0  after token+position embeddings    (64, 768)     49,152        1,024
//     1  after layer 0                      (64, 768)     49,152      197,632
//     2  after layer 1                      (64, 768)     49,152      394,240
//     3  after layer 2                      (64, 768)     49,152      590,848
//     4  after layer 3                      (64, 768)     49,152      787,456
//     5  after layer 4                      (64, 768)     49,152      984,064
//     6  after layer 5                      (64, 768)     49,152    1,180,672
//     7  after layer 6                      (64, 768)     49,152    1,377,280
//     8  after layer 7                      (64, 768)     49,152    1,573,888
//     9  after layer 8                      (64, 768)     49,152    1,770,496
//    10  after layer 9                      (64, 768)     49,152    1,967,104
//    11  after layer 10                     (64, 768)     49,152    2,163,712
//    12  after layer 11                     (64, 768)     49,152    2,360,320
//    13  after the final layernorm          (64, 768)     49,152    2,556,928
//    14  logits at position 63 only         (1, 50304)    50,304    2,753,536
//
// Tensor 0 is llm.c's "encoded"; 1..12 are "residual3[l]", 13 is "lnf"; 14 is "logits".
// Take a look at the model.acts structure in llm.c/train_gpt2.c - this is where the
// activations above come from.
// ** END FILE FORMAT **
#include "reference.hpp"

#include <cassert>
#include <cstdint>
#include <string>

#include "mappedfile.hpp"

using namespace std;

Reference::Reference(const string file) : mp(MappedFile(file.c_str())) {
    constexpr auto OFFSET_MAGIC = 0;                         // Magic number
    constexpr auto OFFSET_VERSION = 1 * sizeof(int32_t);     // Version
    constexpr auto OFFSET_B = 2 * sizeof(int32_t);           // # of batches
    constexpr auto OFFSET_T = 3 * sizeof(int32_t);           // # of token ids
    constexpr auto OFFSET_C = 4 * sizeof(int32_t);           // # of channels
    constexpr auto OFFSET_L = 5 * sizeof(int32_t);           // # of layers
    constexpr auto OFFSET_VP = 6 * sizeof(int32_t);          // Padded vocab size
    constexpr auto OFFSET_NUMTENSORS = 7 * sizeof(int32_t);  // # of tensors dumped
    constexpr auto OFFSET_LOGITROWS = 8 * sizeof(int32_t);   // # of final logit rows

    constexpr auto MAGIC = 20260928;
    constexpr auto VERSION = 1;

    // Read parameters from the file
    assert(mp.to_int32(mp.bytes().subspan(OFFSET_MAGIC, sizeof(int32_t))) == MAGIC);
    assert(mp.to_int32(mp.bytes().subspan(OFFSET_VERSION, sizeof(int32_t))) == VERSION);
    B = mp.to_int32(mp.bytes().subspan(OFFSET_B, sizeof(int32_t)));
    T = mp.to_int32(mp.bytes().subspan(OFFSET_T, sizeof(int32_t)));
    C = mp.to_int32(mp.bytes().subspan(OFFSET_C, sizeof(int32_t)));
    L = mp.to_int32(mp.bytes().subspan(OFFSET_L, sizeof(int32_t)));
    Vp = mp.to_int32(mp.bytes().subspan(OFFSET_VP, sizeof(int32_t)));
    num_tensors = mp.to_int32(mp.bytes().subspan(OFFSET_NUMTENSORS, sizeof(int32_t)));
    logit_rows = mp.to_int32(mp.bytes().subspan(OFFSET_LOGITROWS, sizeof(int32_t)));

    // Calculate locations of activations
    offset_encoded = 1024;
    offset_residual.reserve(L);
    for (int i = 0; i < L; i++) {
        offset_residual[i] = offset_encoded + (T * C) * sizeof(float) + i * (T * C) * sizeof(float);
    }
    offset_lnf = offset_residual[L - 1] + T * C * sizeof(float);
    offset_logits = offset_lnf + T * C * sizeof(float);

    // Sanity checks ... but not really required.
    size_encoded = offset_residual[0] - offset_encoded;
    size_residual.reserve(L);
    for (int i = 0; i < L - 1; i++) {
        size_residual[i] = offset_residual[i + 1] - offset_residual[i];
    }
    size_lnf = offset_logits - offset_lnf;
    size_logits = Vp * sizeof(float);

    // Some sanity asserts since weights are known in advance
    // Check if offset math above is correct
    assert(offset_logits + size_logits == mp.bytes().size());
}

[[nodiscard]] std::span<const float> Reference::spanAtIndex(const int index) const {
    assert(index >= 0 && index <= 14);
    if (index == 0) {
        return mp.span_at<const float>(offset_encoded, C * T);
    } else if (index >= 1 && index <= L) {
        // remember - index 0 is encoded, and the residuals start from index = 1
        return mp.span_at<const float>(offset_residual[index - 1], C * T);
    } else if (index == L + 1) {
        // for the final layernorm, remember that we only care about the last row
        // the decoder slices out the last row and discards everything else
        // the reference file has (64, 768) as the shape, but the decoder only cares about the last
        // row - shape (1, 768)
        return mp.span_at<const float>(offset_lnf + (T - 1) * C * sizeof(float), C);
    } else if (index == L + 2) {
        // Logits for last row only
        return mp.span_at<const float>(offset_logits, Vp);
    }
    std::unreachable();
}