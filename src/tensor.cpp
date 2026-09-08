// tensor.cpp — implementations for tensor.hpp. Definitions go here.
#include "tensor.hpp"

#include <assert.h>

#include <cfloat>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

namespace tn {

Tensor::Tensor(int rows, int cols) : rows_(rows), cols_(cols), row_stride_(cols), col_stride_(1) {
    data_ = std::vector<float>(rows * cols, 0.0f);
}

void Tensor::set(int i, int j, float val) {
    assert(i >= 0 && i < rows_);
    assert(j >= 0 && j < cols_);
    int index = (i * row_stride_) + (j * col_stride_);
    assert(index >= 0 && index < static_cast<int>(data_.size()));
    data_[index] = val;
}

float Tensor::at(int i, int j) const {
    assert(i >= 0 && i < rows_);
    assert(j >= 0 && j < cols_);
    int index = (i * row_stride_) + (j * col_stride_);
    assert(index >= 0 && index < static_cast<int>(data_.size()));
    return data_[index];
}

int Tensor::rows() const { return rows_; }

int Tensor::cols() const { return cols_; }

float Tensor::val() const {
    assert(rows_ == 1 && cols_ == 1);
    return at(0, 0);
}

float Tensor::sum() const {
    float sum = 0.0f;
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            sum += at(i, j);
        }
    }
    return sum;
}

int Tensor::max_idx(int row) const {
    assert(row >= 0 && row < rows_);
    float max = -FLT_MAX;
    int max_idx = 0;
    for (int i = 0; i < cols_; i++) {
        if (at(row, i) > max) {
            max = at(row, i);
            max_idx = i;
        }
    }
    return max_idx;
}

Tensor Tensor::zeros(int rows, int cols) { return Tensor(rows, cols); }

Tensor Tensor::add_bias(const Tensor& bias) const {  // this: (rows, cols), bias: (1, cols)
    assert(bias.cols_ == cols_ && bias.rows_ == 1);
    Tensor output(rows_, cols_);
    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
            output.set(i, j, at(i, j) + bias.at(0, j));
        }
    }
    return output;
}

// input: Tensor(batchsize, logits)
// output: Tensor(batchsize, probabilities)
Tensor Tensor::softmax() const {
    Tensor output = Tensor(rows_, cols_);

    // First find the max of the tensor values and then subtract the max from each
    // value so that the exp does not blow up
    // TODO: exp is being called twice. Cache it.
    for (int row = 0; row < rows_; row++) {
        float row_max = std::numeric_limits<float>::lowest();
        float row_sum = 0.0f;
        for (int col = 0; col < cols_; col++) {
            row_max = (at(row, col) > row_max ? at(row, col) : row_max);
        }
        for (int col = 0; col < cols_; col++) {
            row_sum += std::exp(at(row, col) - row_max);
        }
        for (int col = 0; col < cols_; col++) {
            output.set(row, col, std::exp(at(row, col) - row_max) / row_sum);
        }
    }
    return output;
}

Tensor Tensor::matmul(const Tensor& t) const { /* C = A @ B */
    assert(cols_ == t.rows_);
    Tensor output(rows_, t.cols_);

    for (int p = 0; p < rows_; p++) {
        for (int q = 0; q < t.cols_; q++) {
            float val = 0.0f;
            for (int k = 0; k < cols_; k++) {
                val += (at(p, k) * t.at(k, q));
            }
            output.set(p, q, val);
        }
    }
    return output;
}

Tensor Tensor::clone() const { return Tensor(rows_, cols_, row_stride_, col_stride_, data_); }

Tensor Tensor::transpose() const {
    Tensor t;
    t.data_ = data_;
    t.rows_ = cols_;
    t.cols_ = rows_;
    t.row_stride_ = col_stride_;
    t.col_stride_ = row_stride_;
    return t;
}

std::vector<float> Tensor::flatten() const {
    std::vector<float> output;
    output.reserve(rows_ * cols_);
    if (row_stride_ == cols_ && col_stride_ == 1)
        output = data_;
    else {
        for (int i = 0; i < rows_; i++) {
            for (int j = 0; j < cols_; j++) {
                output.push_back(at(i, j));
            }
        }
    }
    return output;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << std::fixed << std::setprecision(5);
    for (int i = 0; i < t.rows(); ++i) {
        for (int j = 0; j < t.cols(); ++j) {
            os << std::setw(10) << t.at(i, j);
        }
        os << std::endl;
    }
    return os;
}

}  // namespace tn
