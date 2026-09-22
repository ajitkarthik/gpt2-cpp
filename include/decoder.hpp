#pragma once

#include <limits.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>

#include "tensor.hpp"

using tn::MatrixView;
using tn::Tensor;

namespace decoder {

// y = x@w_T + b
// w (ouptut feature, input feature)
// b (output feature) - b is broadcasted
// x (batch dim, input feature)
// y (batch dim, output feature)
class Linear {
   public:
    Linear(MatrixView w, std::optional<std::span<const float>> b) : w_(w), b_(b) {}
    Linear() = default;
    Tensor forward(const Tensor& x) const;

   private:
    MatrixView w_;
    std::optional<std::span<const float>> b_;
};

// input matrix (A, B)
// output matrix with values in dimension B normed.
// See: https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html
class LayerNorm {
   public:
    LayerNorm() = default;
    LayerNorm(std::span<const float> w, std::span<const float> b, float eps)
        : w_(w), b_(b), eps_(eps) {}
    Tensor forward(const Tensor& x) const;

   private:
    std::span<const float> w_;
    std::span<const float> b_;
    float eps_;
};

// Multi-head self-attention
// input: Shape (T, C)
// output: Shape (T, C)
// T = tokens, C = embedding dimensions
class MHSA {
   public:
    MHSA(std::span<const float> qkvw, std::span<const float> qkvb, std::span<const float> attprojw,
         std::span<const float> attprojb, int nheads)
        : qkvw_(qkvw), qkvb_(qkvb), attprojw_(attprojw), attprojb_(attprojb), nheads_(nheads) {}
    MHSA() = default;
    Tensor forward(const Tensor& x) const;

   private:
    std::span<const float> qkvw_;      // shape (3C, C) C is the embedding dimension
    std::span<const float> qkvb_;      // shape (3C)
    std::span<const float> attprojw_;  // shape (C, C)
    std::span<const float> attprojb_;  // shape (C)
    int nheads_;                       // number of attention heads
};

// Linear (input, hidden), GELU, Linear (hidden, output)
class FFN {
   public:
    FFN(MatrixView w1, std::span<const float> b1, MatrixView wproj, std::span<const float> bproj)
        : l1_(w1, b1), l2_(wproj, bproj) {}
    FFN() = default;
    Tensor forward(const Tensor& x) const;

   private:
    Linear l1_;
    Linear l2_;
};

class Layer {
   public:
    Layer() = default;
    Layer(std::span<const float> ln1w, std::span<const float> ln1b, std::span<const float> qkvw,
          std::span<const float> qkvb, std::span<const float> attprojw,
          std::span<const float> attprojb, std::span<const float> ln2w, std::span<const float> ln2b,
          std::span<const float> fcw, std::span<const float> fcb, std::span<const float> fcprojw,
          std::span<const float> fcprojb, float eps, int nheads, int embsz)
        : ln1_(LayerNorm(ln1w, ln1b, eps)),
          attn_(MHSA(qkvw, qkvb, attprojw, attprojb, nheads)),
          ln2_(LayerNorm(ln2w, ln2b, eps)),
          ffn_(FFN(MatrixView(fcw.data(), 4 * embsz, embsz), fcb,
                   MatrixView(fcprojw.data(), embsz, 4 * embsz), fcprojb)) {}
    Tensor forward(const Tensor& x);

   private:
    LayerNorm ln1_;
    MHSA attn_;
    LayerNorm ln2_;
    FFN ffn_;
};

class Decoder {
   public:
    // Create the transformer
    Decoder(int vocab, int nlayers, int numheads, int embedsize, int vocab_padded)
        : vocab_(vocab),
          nlayers_(nlayers),
          numheads_(numheads),
          embedsize_(embedsize),
          vocab_padded_(vocab_padded) {
        // Instantiate all the layers
        layers_.reserve(nlayers);
        eps_ = 1.0e-5;
    }
    void loadLayerWeights(int i, std::span<const float> ln1w, std::span<const float> ln1b,
                          std::span<const float> qkvw, std::span<const float> qkvb,
                          std::span<const float> attprojw, std::span<const float> attprojb,
                          std::span<const float> ln2w, std::span<const float> ln2b,
                          std::span<const float> fcw, std::span<const float> fcb,
                          std::span<const float> fcprojw, std::span<const float> fcprojb);
    void loadLNWeights(std::span<const float> lnfw, std::span<const float> lnfb);
    void loadVocabProjWeights(std::span<const float> wte);
    Tensor forward(Tensor& x);

   private:
    int vocab_;  // vocabulary size
    int nlayers_;
    int numheads_;
    int embedsize_;
    int vocab_padded_;
    std::vector<Layer> layers_;
    LayerNorm finalLN_;  // Final layer norm, outside all the 12 layers
    Linear vocabproj_;   // Vocabulary projection
    float eps_;
};

}  // namespace decoder