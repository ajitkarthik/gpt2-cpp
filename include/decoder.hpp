#pragma once

#include <limits.h>

#include <cassert>
#include <cmath>
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
    Linear(tn::MatrixView w, std::optional<std::span<const float>> b) : w_(w), b_(b) {
        if (b_.has_value()) {
            assert(static_cast<int>((*b_).size()) == w.rows);
        }
    }

    Tensor forward(const Tensor& x) const {
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
    MHSA(std::span<const float> qkvw, std::span<const float> attprojw, std::span<const float> qkvb,
         std::span<const float> attprojb, int nheads)
        : qkvw_(qkvw), qkvb_(qkvb), attprojw_(attprojw), attprojb_(attprojb), nheads_(nheads) {}
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
        : w1_(w1), b1_(b1), wproj_(wproj), bproj_(bproj) {}
    Tensor forward(const Tensor& x) const {
        Linear l1(w1_, b1_);
        Linear l2(wproj_, bproj_);
        Tensor h = l1.forward(x);
        gelu_(h);
        return l2.forward(h);
    }

   private:
    MatrixView w1_;
    std::span<const float> b1_;
    MatrixView wproj_;
    std::span<const float> bproj_;
};

class Layer {
   public:
   private:
};

class Decoder {
   public:
    int contextSize;
    int vocab;
    int layers;
    int numheads;
    int embedsize;
    int vocab_padded;

    // Read the

    // Create the transformer
    Decoder(int contextSize, int vocab, int layers, int numheads, int embedsize, int vocab_padded)
        : contextSize(contextSize),
          vocab(vocab),
          layers(layers),
          numheads(numheads),
          embedsize(embedsize),
          vocab_padded(vocab_padded) {}
};

}  // namespace decoder