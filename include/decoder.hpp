#pragma once
#include <sys/stat.h>

#include <cassert>
#include <cmath>
#include <span>

#include "tensor.hpp"

using tn::MatrixView;
using tn::Tensor;
using namespace std;

namespace decoder {

// See https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html
inline Tensor gelu_(Tensor& x) {
    constexpr float k = 0.7978845608028654f;  // sqrt(2/pi)
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            const float v = x.at(i, j);
            const float inner = k * (v + 0.044715f * v * v * v);
            x.set(i, j, 0.5f * v * (1.0f + tanhf(inner)));
        }
    }
    return x;
}

// y = x@w_T + b
// w (ouptut feature, input feature)
// b (output feature) - b is broadcasted
// x (batch dim, input feature)
// y (batch dim, output feature)
class Linear {
   public:
    Linear(MatrixView w, span<const float> b) : w_(w), b_(b) {
        assert(static_cast<int>(b.size()) == w.rows);
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
                y.set(i, j, val + b_[j]);
            }
        }
        return y;
    }

   private:
    MatrixView w_;
    span<const float> b_;
};

// input matrix (A, B)
// output matrix with values in dimension B normed.
// See: https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html
class LayerNorm {
   public:
    LayerNorm(span<const float> w, span<const float> b, float eps) : w_(w), b_(b), eps_(eps) {}
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
    span<const float> w_;
    span<const float> b_;
    float eps_;
};

// Multi-head self-attention
// input: Shape (T, C)
// output: Shape (T, C)
// T = tokens, C = embedding dimensions
class MHSA {
   public:
    MHSA(span<const float> qkvw, span<const float> attprojw, span<const float> qkvb,
         span<const float> attprojb, int nheads)
        : qkvw_(qkvw), qkvb_(qkvb), attprojw_(attprojw), attprojb_(attprojb), nheads_(nheads) {}
    Tensor forward(const Tensor& x) const {
        int C = x.cols();  // For gpt-2 this is 768
        int T = x.rows();
        Tensor out(T, C);    // Shape (T, C)
        Tensor out_t(C, T);  // Shape (C, T)

        // first step is the linear: h = x @ qkvw^T + qkvb
        // qkvw is of shape (3C, C).
        // shape: q (T, 3C) <- x (T, C) @ qkvw (3C, C)^T + qkvb (3C)
        MatrixView qkvw(&qkvw_[0], 3 * C, C);

        // Now apply the linear
        Linear lqkv(qkvw, qkvb_);
        Tensor qkv = lqkv.forward(x);  // Shape (T, 3C)
        assert(qkv.rows() == T && qkv.cols() == 3 * C);

        // next step is to separate the qkv tensor shape = (T, 3C)
        // into 3 tensors q, k, v each shape = (T, C)
        // Also, we need to split each q, k, v further into shape (T, C/nheads) for each of the
        // attention heads. Since we only have a 2-D tensor library, run a for loop nheads times
        // with each q, k, v matrix of shape = (T, C/nheads)
        // we'll do this with a 0-copy MatrixView that offsets cleverly into the qkv tensor
        assert(C % nheads_ == 0);
        for (int i = 0; i < nheads_; i++) {
            // Per head q, k, and v. Shape (T, C/nheads)
            MatrixView qh(qkv.data().data() + (i * C) / nheads_, T, C / nheads_, 3 * C, 1);
            MatrixView kh(qkv.data().data() + C + (i * C) / nheads_, T, C / nheads_, 3 * C, 1);
            MatrixView vh(qkv.data().data() + (2 * C) + (i * C) / nheads_, T, C / nheads_, 3 * C,
                          1);
            // Shape (C/nheads, T)
            MatrixView kht = kh.transpose();
            // scaled dot product (T, C/nheads) @ (C/nheads, T) -> (T, T)
            Tensor sdp = matmul(qh, kht).scale(1.0 / sqrt(C / nheads_));  // Shape (T, T)
            assert(sdp.rows() == T && sdp.cols() == T);
            // Softmax
            Tensor softmax = sdp.softmax();  // Shape (T, T)
            // Extract the value vector
            Tensor val = softmax.matmul(vh);  // Shape (T, T) @ (T, C/nheads) -> (T, C/nheads)
            assert(val.rows() == T && val.cols() == C / nheads_);
            // Stack the matrix from each head to get (T, C)
            // To do this, we transpose val to get (C/nheads, T)
            // then stack to get (C, T)
            // then transpose again to get (T, C)
            Tensor val_t = val.transpose();  // val for head i, shape (C/nheads, T)
            out_t.data().reserve(out.data().size() + val_t.data().size());
            out_t.data().insert(out.data().end(), make_move_iterator(val_t.data().begin()),
                                make_move_iterator(val_t.data().end()));
        }
        // At the end of the loop we have a tensor that's (C, T)
        // Transpose it to get (T, C)
        out = out_t.transpose();
        assert(out.rows() == T && out.cols() == C);
        // Apply the linear projection
        // shapes: out(T, C) @ attproj(C, C) -> (T, C)
        Linear lproj(MatrixView(&attprojw_[0], C, C), attprojb_);
        Tensor attnproj = lproj.forward(out);
        assert(attnproj.rows() == T && attnproj.cols() == C);
        return attnproj;
    }

   private:
    span<const float> qkvw_;      // shape (3C, C) C is the embedding dimension
    span<const float> qkvb_;      // shape (3C)
    span<const float> attprojw_;  // shape (C, C)
    span<const float> attprojb_;  // shape (C)
    int nheads_;                  // number of attention heads
};

// Linear (input, hidden), GELU, Linear (hidden, output)
class FFN {
   public:
    FFN(MatrixView w1, span<const float> b1, MatrixView wproj, span<const float> bproj)
        : w1_(w1), b1_(b1), wproj_(wproj), bproj_(bproj) {}
    Tensor forward(const Tensor& x) const {
        Linear l1(w1_, b1_);
        Linear l2(wproj_, bproj_);
        Tensor h = l1.forward(x);
        return l2.forward(gelu_(h));
    }

   private:
    MatrixView w1_;
    span<const float> b1_;
    MatrixView wproj_;
    span<const float> bproj_;
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