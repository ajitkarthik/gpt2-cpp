#pragma once
#include <cassert>
#include <cmath>
#include <span>

#include "tensor.hpp"

using tn::Tensor;

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
            x.set(i, j, 0.5f * v * (1.0f + std::tanhf(inner)));
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
    Linear(MatrixView w, std::span<const float> b) : w_(w), b_(b) {
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
    std::span<const float> b_;
};

class LayerNorm {
   public:
   private:
};

class MHSA {
   public:
   private:
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
        return l2.forward(gelu_(h));
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