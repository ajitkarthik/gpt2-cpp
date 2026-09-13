#pragma once
#include <sys/stat.h>

#include <cassert>
#include <cmath>
#include <span>

#include "tensor.hpp"

using tn::Tensor;
using namespace std;

namespace decoder {

struct MatrixView {
    const float* data;
    int rows, cols;
    int row_stride, col_stride;
    MatrixView(const float* data, int rows, int cols)
        : data(data), rows(rows), cols(cols), row_stride(cols), col_stride(1) {}
    MatrixView(const float* data, int rows, int cols, int row_stride, int col_stride)
        : data(data), rows(rows), cols(cols), row_stride(row_stride), col_stride(col_stride) {}
    float at(int i, int j) const {
        assert(i < rows && j < cols);
        return *(data + i * row_stride + j * col_stride);
    }
};

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
class MHSA {
   public:
    MHSA(span<const float> qkvw, span<const float> attprojw, span<const float> qkvb,
         span<const float> attprojb, int embsz)
        : qkvw_(qkvw), qkvb_(qkvb), attprojw_(attprojw), attprojb_(attprojb), embsz_(embsz) {}
    Tensor forward(const Tensor& x) const {
        // first step is the linear: h = x @ qkvw^T + qkvb
        // qkvw is of shape (3C, C). Let's first slice it into qw, kw, vw each of (C, C)
        // This is better than post-slicing after the linear, so that we do don't have to hold
        // 1 contiguous (T, 3C) tensor in memory. Instead we have 3 different (T, C) tensors
        MatrixView qw(&qkvw_[0], embsz_, embsz_, embsz_,
                      1);  // data, rows, cols, row stride, col stride
        MatrixView kw(&qkvw_[embsz_ * embsz_ * sizeof(float)], embsz_, embsz_, embsz_,
                      1);  // data, rows, cols, row stride, col stride
        MatrixView vw(&qkvw_[2 * embsz_ * embsz_ * sizeof(float)], embsz_, embsz_, embsz_,
                      1);  // data, rows, cols, row stride, col stride
        // Now apply the linear
        // q = x @ qw^T + b
        // shape: q (T, C) <- x (T, C) @ qw (C, C)^T + b (C)
        Linear l1q(qw, qkvb_.subspan(0, embsz_));
        Linear l1k(kw, qkvb_.subspan(embsz_, embsz_));
        Linear l1v(vw, qkvb_.subspan(2 * embsz_, embsz_));
        Tensor q = l1q.forward(x);  // Shape (T, C)
        Tensor k = l1k.forward(x);  // Shape (T, C)
        Tensor v = l1v.forward(x);  // Shape (T, C)
    }

   private:
    span<const float> qkvw_;      // shape (3C, C) C=embsz
    span<const float> qkvb_;      // shape (3C)
    span<const float> attprojw_;  // shape (C, C)
    span<const float> attprojb_;  // shape (C)
    int embsz_;                   // embedding dimensions
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