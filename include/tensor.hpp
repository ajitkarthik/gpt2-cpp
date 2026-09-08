// tensor.hpp — a tiny reverse-mode autograd tensor.
//
// Design: value-semantics handle (Tensor) over a shared graph node
// (Tensor::Node). Ops record their inputs and a local backward closure that
// pushes gradients onto its inputs' `grad` buffers. `backward()` seeds the
// output gradient, walks the graph in reverse topological order, and runs each
// closure once. Data is a flat row-major buffer described by `shape`.
#pragma once

#include <functional>
#include <memory>
#include <ostream>
#include <stack>
#include <unordered_set>
#include <vector>

namespace tn {
class Tensor {
   private:
    enum class OpCode { NONE, MATMUL };
    struct Node {
        std::vector<float> data;
        std::vector<float> grad;
        int rows;
        int cols;
        int row_stride;
        int col_stride;
        OpCode opcode;
        // track the computational graph for backprop
        std::vector<std::shared_ptr<Node>> prev;
        // local callback function for backprop
        std::function<void()> backward;

        Node(int rows, int cols, float fill)
            : data(rows * cols, fill),
              grad(rows * cols, 0.0f),
              rows(rows),
              cols(cols),
              row_stride(cols),
              col_stride(1),
              opcode(OpCode::NONE) {}

        Node(int rows, int cols, std::vector<float> data)
            : data(std::move(data)),
              grad(rows * cols,
                   0.0f),  // can't use data.size() because of the move operation
              rows(rows),
              cols(cols),
              row_stride(cols),
              col_stride(1),
              opcode(OpCode::NONE) {}

        Node(int rows, int cols, int row_stride, int col_stride, std::vector<float> data)
            : data(std::move(data)),
              grad(rows * cols,
                   0.0f),  // can't use data.size() because of the move operation
              rows(rows),
              cols(cols),
              row_stride(row_stride),
              col_stride(col_stride),
              opcode(OpCode::NONE) {}
    };

    // utility function
    void DFSVisit(Tensor::Node* curr, std::unordered_set<Node*>& visited, std::stack<Node*>& stack);

   public:
    // Empty/undefined tensor (no storage). Assign into it later.
    Tensor() = default;

    // Allocate a rows x cols tensor, elements zero-initialized.
    Tensor(int rows, int cols);

    // Allocate a rows x cols tensor with every element set to `fill`.
    Tensor(int rows, int cols, float fill);

    // Wrap existing row-major data; data.size() must equal rows * cols.
    Tensor(int rows, int cols, std::vector<float> data);

    // Given a node, get it's corresponding Tensor handle
    Tensor(Node& n);

    static Tensor randn(int rows, int cols);
    static Tensor zeros(int rows, int cols);
    // Reseed the generator backing randn(). By default it is seeded from
    // std::random_device; call this to make a run reproducible.
    static void seed(uint32_t s);

    // misc helpers
    // set element [i, j]
    void set(int i, int j, float val);
    // return element [i, j]
    float at(int i, int j) const;
    // return grad at [i, j]
    float grad_at(int i, int j) const;
    // zero out the grad
    void zero_grad();
    // number of rows and cols
    int rows() const;
    int cols() const;
    // used for the test pass
    Tensor softmax() const;
    // find index of max element in given row of tensor
    int max_idx(int row);
    // returns the value of a tensor. Makes sense only for a tensor with 1 value
    float val() const;
    // flatten a Tensor to a vector<float>
    std::vector<float> flatten() const;

    // backward pass
    void backward(void);
    void adjust_weights(float lr);

    // Tensor ops
    Tensor transpose() const;
    Tensor matmul(const Tensor&) const;
    Tensor clone() const;
    float sum() const;                          // returns the sum of all elements of a tensor
    Tensor add_bias(const Tensor& bias) const;  // this: (rows, cols), bias: (1, cols)
    Tensor relu() const;
    Tensor fused_cross_entropy_loss(Tensor labels) const;

    std::shared_ptr<Node> node_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);
}  // namespace tn
