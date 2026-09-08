// tensor.hpp — only forward mode tensor library
#pragma once

#include <ostream>
#include <vector>

namespace tn {
class Tensor {
   private:
    std::vector<float> data_;
    int rows_;
    int cols_;
    int row_stride_;
    int col_stride_;

   public:
    // Constructor for a tensor with a given fill value
    Tensor(int rows, int cols, float fill)
        : data_(rows * cols, fill), rows_(rows), cols_(cols), row_stride_(cols), col_stride_(1) {}

    // Constructor for a tensor with a given data vector
    Tensor(int rows, int cols, std::vector<float> data)
        : data_(std::move(data)), rows_(rows), cols_(cols), row_stride_(cols), col_stride_(1) {}

    // Constructor for a tensor with a given data vector and row/col strides
    Tensor(int rows, int cols, int row_stride, int col_stride, std::vector<float> data)
        : data_(std::move(data)),
          rows_(rows),
          cols_(cols),
          row_stride_(row_stride),
          col_stride_(col_stride) {}

    // Empty/undefined tensor (no storage). Assign into it later.
    Tensor() : rows_(0), cols_(0), row_stride_(0), col_stride_(0) {}

    // Allocate a rows x cols tensor, elements zero-initialized.
    Tensor(int rows, int cols);

    static Tensor zeros(int rows, int cols);

    // misc helpers
    // set element [i, j]
    void set(int i, int j, float val);
    // return element [i, j]
    float at(int i, int j) const;
    // number of rows and cols
    int rows() const;
    int cols() const;
    // used for the test pass
    Tensor softmax() const;
    // find index of max element in given row of tensor
    int max_idx(int row) const;
    // returns the value of a tensor. Makes sense only for a tensor with 1 value
    float val() const;
    // flatten a Tensor to a vector<float>
    std::vector<float> flatten() const;

    // Tensor ops
    Tensor transpose() const;
    Tensor matmul(const Tensor&) const;
    Tensor clone() const;
    float sum() const;                          // returns the sum of all elements of a tensor
    Tensor add_bias(const Tensor& bias) const;  // this: (rows, cols), bias: (1, cols)
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);
}  // namespace tn
