#pragma once

#include <limits.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <optional>
#include <span>

#include "tensor.hpp"

using tn::MatrixView;
using tn::Tensor;

namespace decoder {

// See https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html
inline void gelu_(Tensor& x) {
    constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            const float v = x.at(i, j);
            const float inner = k * (v + 0.044715f * v * v * v);
            x.set(i, j, 0.5f * v * (1.0f + tanhf(inner)));
        }
    }
}

// y = x@w_T + b
// w (ouptut feature, input feature)
// b (output feature) - b is broadcasted
// x (batch dim, input feature)
// y (batch dim, output feature)
class Linear {
   public:
    Linear(MatrixView w, std::optional<std::span<const float>> b) : w_(w), b_(b) {}
    Linear() = default;

    Tensor forward(const Tensor& x) const {
        if (b_.has_value()) {
            assert(static_cast<int>((*b_).size()) == w_.rows);
        }

        assert(x.cols() == w_.cols);
        Tensor y(x.rows(), w_.rows);
        for (int i = 0; i < x.rows(); i++) {
            for (int j = 0; j < w_.rows; j++) {
                float val = 0.0f;
                for (int k = 0; k < x.cols(); k++) {
                    val += (x.at(i, k) * w_.at(j, k));
                }
                if (b_.has_value()) {
                    y.set(i, j, val + (*b_)[j]);
                }
            }
        }
        return y;
    }

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
    Tensor forward(const Tensor& x) const {
        assert(x.cols() == static_cast<int>(w_.size()));
        assert(x.cols() == static_cast<int>(b_.size()));
        Tensor out(x.rows(), x.cols());
        for (int i = 0; i < x.rows(); i++) {
            float mu = 0, var = 0;
            for (int j = 0; j < x.cols(); j++) {
                mu += x.at(i, j);
            }
            mu /= x.cols();
            for (int j = 0; j < x.cols(); j++) {
                var += (x.at(i, j) - mu) * (x.at(i, j) - mu);
            }
            var /= x.cols();  // Var(x) = E[(X - mu)^2], mu = E[x]
            float s = sqrt(var + eps_);
            for (int j = 0; j < x.cols(); j++) {
                out.set(i, j, (((x.at(i, j) - mu) / s) * w_[j]) + b_[j]);
            }
        }
        return out;
    }

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
    Tensor forward(const Tensor& x) const {
        int C = x.cols();  // For gpt-2 this is 768
        int T = x.rows();
        Tensor out(T, C);  // Shape (T, C)

        // Check tensor shapes for sanity
        assert(qkvw_.size() == static_cast<size_t>(C * 3 * C));
        assert(qkvb_.size() == static_cast<size_t>(3 * C));
        assert(attprojw_.size() == static_cast<size_t>(C * C));
        assert(attprojb_.size() == static_cast<size_t>(C));

        // first step is the linear: h = x @ qkvw^T + qkvb
        // qkvw is of shape (3C, C).
        // shape: lqkv (T, 3C) <- x (T, C) @ qkvw (3C, C)^T + qkvb (3C)
        MatrixView qkvw(&qkvw_[0], 3 * C, C);

        // Now apply the linear
        Linear lqkv(qkvw, qkvb_);
        Tensor qkv = lqkv.forward(x);  // Shape (T, 3C)
        assert(qkv.rows() == T && qkv.cols() == 3 * C);

        // next step is to separate the qkv Tensor shape (T, 3C)
        // into 3 Tensors q, k, v each shape = (T, C)
        // Also, we need to split each q, k, v further into shape (T, C/nheads) for each of the
        // attention heads. Since we only have a 2-D Tensor library, run a for loop nheads times
        // with each q, k, v matrix of shape = (T, C/nheads)
        // we'll do this with a 0-copy MatrixView that offsets cleverly into the qkv Tensor
        assert(C % nheads_ == 0);
        for (int i = 0; i < nheads_; i++) {
            // Per head q, k, and v. Shape (T, C/nheads)
            MatrixView qh(qkv.raw() + (i * C) / nheads_, T, C / nheads_, 3 * C, 1);
            MatrixView kh(qkv.raw() + C + (i * C) / nheads_, T, C / nheads_, 3 * C, 1);
            MatrixView vh(qkv.raw() + (2 * C) + (i * C) / nheads_, T, C / nheads_, 3 * C, 1);
            // Shape (C/nheads, T)
            MatrixView kht = kh.transpose();
            // scaled dot product (T, C/nheads) @ (C/nheads, T) -> (T, T)
            Tensor sdp = matmul(qh, kht).scale(1.0 / sqrt(C / nheads_));  // Shape (T, T)
            assert(sdp.rows() == T && sdp.cols() == T);
            // Apply causal mask - upper triangular entries go to -INF before softmax
            for (int sdp_row = 0; sdp_row < sdp.rows(); sdp_row++) {
                for (int sdp_col = 0; sdp_col < sdp.cols(); sdp_col++) {
                    if (sdp_col > sdp_row) {
                        sdp.set(sdp_row, sdp_col, -std::numeric_limits<float>::infinity());
                    }
                }
            }
            // Softmax
            Tensor softmax = sdp.softmax();  // Shape (T, T)
            // Extract the value vector
            Tensor val = softmax.matmul(vh);  // Shape (T, T) @ (T, C/nheads) -> (T, C/nheads)
            assert(val.rows() == T && val.cols() == C / nheads_);
            // Stack the matrix from each head to get (T, C)
            for (int t = 0; t < T; t++) {
                memcpy(out.raw() + (t * C) + (i * C) / nheads_, val.raw() + t * (C / nheads_),
                       C / nheads_ * sizeof(float));
            }
        }
        // Apply the linear projection
        // shapes: out(T, C) @ attproj(C, C) -> (T, C)
        Linear lproj(MatrixView(&attprojw_[0], C, C), attprojb_);
        Tensor attnproj = lproj.forward(out);
        assert(attnproj.rows() == T && attnproj.cols() == C);
        return attnproj;
    }

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
    Tensor forward(const Tensor& x) const {
        Tensor h = l1_.forward(x);
        gelu_(h);
        return l2_.forward(h);
    }

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
    Tensor forward(const Tensor& x) {  // shape (T, C)
        Tensor out(x.rows(), x.cols());
        out = ln1_.forward(x);
        out = attn_.forward(out);
        out = out + x;  // add the residual stream
        out = ln2_.forward(out);
        out = ffn_.forward(out);
        out = out + x;  // add the residual stream
        assert(out.rows() == x.rows() && out.cols() == x.cols());
        return out;
    }

   private:
    LayerNorm ln1_;
    MHSA attn_;
    LayerNorm ln2_;
    FFN ffn_;
};

class Decoder {
   public:
    // Create the transformer
    Decoder(int contextSize, int vocab, int nlayers, int numheads, int embedsize, int vocab_padded)
        : contextSize_(contextSize),
          vocab_(vocab),
          nlayers_(nlayers),
          numheads_(numheads),
          embedsize_(embedsize),
          vocab_padded_(vocab_padded) {
        // Instantiate all the layers
        layers_.reserve(nlayers);
        eps_ = 1.0e-5;
    }
    // Loads weights for a layer
    void loadLayerWeights(int i, std::span<const float> ln1w, std::span<const float> ln1b,
                          std::span<const float> qkvw, std::span<const float> qkvb,
                          std::span<const float> attprojw, std::span<const float> attprojb,
                          std::span<const float> ln2w, std::span<const float> ln2b,
                          std::span<const float> fcw, std::span<const float> fcb,
                          std::span<const float> fcprojw, std::span<const float> fcprojb) {
        layers_[i] = Layer(ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb,
                           fcprojw, fcprojb, eps_, numheads_, embedsize_);
    }

    void loadLNWeights(std::span<const float> lnfw, std::span<const float> lnfb) {
        finalLN_ = LayerNorm(lnfw, lnfb, eps_);
    }

    void loadVocabProjWeights(std::span<const float> wte) {
        vocabproj_ = Linear(MatrixView(wte.data(), vocab_padded_, embedsize_), std::nullopt);
    }

    // The decoder expects x to be already embedded and positionally encoded
    // The input into the decoder is of shape (T, C), and output are logits of shape (1, V)
    // the predicted token
    Tensor forward(Tensor& x) {
        assert(x.cols() == embedsize_);  // shape of x (T, C)
        for (int i = 0; i < nlayers_; i++) {
            x = layers_[i].forward(x);
        }
        // Just grab the last position so we'll end up with (1, C)
        Tensor logits = Tensor(1, embedsize_);
        for (int i = 0; i < embedsize_; i++) {
            logits.set(0, i, x.at(x.rows() - 1, i));
        }

        logits = finalLN_.forward(logits);    // Final layerNorm, shape (1, C)
        logits = vocabproj_.forward(logits);  // Vocabulary projection, shape (1, Vp)
        // Remove the padded entries
        Tensor final_logits(1, vocab_);
        for (int i = 0; i < vocab_; i++) {
            final_logits.set(0, i, logits.at(0, i));
        }
        assert(final_logits.rows() == 1 && final_logits.cols() == vocab_);
        return final_logits;
    }

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