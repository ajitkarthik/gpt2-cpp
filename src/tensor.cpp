// tensor.cpp — implementations for tensor.hpp. Definitions go here.
#include "tensor.hpp"

#include <assert.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <stack>
#include <unordered_set>
#include <vector>

namespace tn {

Tensor::Tensor(int rows, int cols, float fill) { node_ = std::make_shared<Node>(rows, cols, fill); }

Tensor::Tensor(int rows, int cols) { node_ = std::make_shared<Node>(rows, cols, 0.0f); }

void Tensor::set(int i, int j, float val) {
    assert(i >= 0 && i < rows());
    assert(j >= 0 && j < cols());
    node_->data[(i * node_->row_stride) + (j * node_->col_stride)] = val;
}

float Tensor::at(int i, int j) const {
    assert(i >= 0 && i < rows());
    assert(j >= 0 && j < cols());
    return node_->data[(i * node_->row_stride) + (j * node_->col_stride)];
}

float Tensor::grad_at(int i, int j) const {
    assert(i >= 0 && i < rows());
    assert(j >= 0 && j < cols());
    return node_->grad[(i * node_->row_stride) + (j * node_->col_stride)];
}

void Tensor::zero_grad() { std::fill(node_->grad.begin(), node_->grad.end(), 0.0f); }

int Tensor::rows() const { return node_->rows; }

int Tensor::cols() const { return node_->cols; }

float Tensor::val() const {
    assert(rows() == 1 && cols() == 1);
    return at(0, 0);
}

float Tensor::sum() const {
    float sum = 0.0f;
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) {
            sum += at(i, j);
        }
    }
    return sum;
}

int Tensor::max_idx(int row) {
    assert(row >= 0 && row < rows());
    float max = FLT_MIN;
    int max_idx = 0;
    for (int i = 0; i < cols(); i++) {
        if (at(row, i) > max) {
            max = at(row, i);
            max_idx = i;
        }
    }
    return max_idx;
}

Tensor Tensor::zeros(int rows, int cols) { return Tensor(rows, cols); }

static std::mt19937& randn_gen() {
    static std::mt19937 gen{std::random_device{}()};
    return gen;
}

void Tensor::seed(uint32_t s) { randn_gen().seed(s); }

Tensor Tensor::randn(int rows, int cols) {
    std::normal_distribution<float> dist(0.0f, 1.0f);  // mean 0, stddev 1

    Tensor t(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) t.set(i, j, dist(randn_gen()));
    return t;
}

Tensor::Tensor(int rows, int cols, std::vector<float> data) {
    assert(static_cast<int>(data.size()) == rows * cols);
    node_ = std::make_shared<Node>(rows, cols, std::move(data));
}

Tensor::Tensor(Node& n) { node_ = std::make_shared<Node>(n); }

Tensor Tensor::add_bias(const Tensor& bias) const {  // this: (rows, cols), bias: (1, cols)
    assert(bias.cols() == cols() && bias.rows() == 1);
    Tensor output(rows(), cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) {
            output.set(i, j, at(i, j) + bias.at(0, j));
        }
    }

    // Save away for the backward pass
    auto a = node_;
    auto b = bias.node_;
    /* Note: here we save the raw pointer otherwise we get a circular ref with
     * shared pointers */
    auto out = output.node_.get();
    output.node_->backward = [a, b, out]() {
        Tensor dOut = Tensor(out->rows, out->cols, out->grad);
        for (int i = 0; i < dOut.rows(); i++)
            for (int j = 0; j < dOut.cols(); j++)
                a->grad[i * a->row_stride + j * a->col_stride] += dOut.at(i, j);

        for (int i = 0; i < dOut.rows(); i++) {
            for (int j = 0; j < dOut.cols(); j++) {
                b->grad[0 * a->row_stride + j * a->col_stride] += dOut.at(i, j);
            }
        }
    };
    output.node_->prev.push_back(a);
    output.node_->prev.push_back(b);
    return output;
}

Tensor Tensor::relu() const {
    Tensor output = Tensor(rows(), cols());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < cols(); j++) {
            if (at(i, j) > 0.0f)
                output.set(i, j, at(i, j));
            else
                output.set(i, j, 0.0f);
        }
    }

    auto a = node_;
    auto out = output.node_.get();
    output.node_->backward = [a, out]() {
        Tensor dOut = Tensor(out->rows, out->cols, out->grad);
        for (int i = 0; i < dOut.rows(); i++) {
            for (int j = 0; j < dOut.cols(); j++) {
                if (a->data[i * a->row_stride + j * a->col_stride] > 0.0f)
                    a->grad[i * a->row_stride + j * a->col_stride] += dOut.at(i, j);
            }
        }
    };
    output.node_->prev.push_back(a);
    return output;
}

// Fused cross-entropy loss
// input: logits: Tensor(batchsize, num_classes), labels: Tensor(batchsize, 1)
// output: loss (Tensor(1,1))
Tensor Tensor::fused_cross_entropy_loss(Tensor labels) const {
    assert(labels.cols() == 1);
    assert(rows() == labels.rows());
    Tensor output = Tensor(1, 1);
    float scalar_loss = 0.0f;

    // compute probabilities
    Tensor probs = this->softmax();

    // negative log liklehood
    for (int row = 0; row < rows(); row++) {
        for (int col = 0; col < cols(); col++) {
            if (col == labels.at(row, 0)) {
                scalar_loss += -std::log(probs.at(row, col));
            }
        }
    }

    output.set(0, 0, scalar_loss / rows());

    auto a = node_;         // logits
    auto b = probs.node_;   // probs
    auto c = labels.node_;  // labels
    output.node_->backward = [a, b, c]() {
        for (int i = 0; i < a->rows; i++) {
            for (int j = 0; j < a->cols; j++) {
                if (j == (c->data[i * c->row_stride]))
                    // Divide by the batchsize here, since scalar loss computed is
                    // across the entire batch
                    a->grad[i * a->row_stride + j * a->col_stride] +=
                        (b->data[i * b->row_stride + j * b->col_stride] - 1) / a->rows;
                else
                    a->grad[i * a->row_stride + j * a->col_stride] +=
                        b->data[i * b->row_stride + j * b->col_stride] / a->rows;
            }
        }
    };

    output.node_->prev.push_back(a);
    return output;
}

// input: Tensor(batchsize, logits)
// output: Tensor(batchsize, probabilities)
Tensor Tensor::softmax() const {
    Tensor output = Tensor(rows(), cols());

    // First find the max of the tensor values and then subtract the max from each
    // value so that the exp does not blow up
    for (int row = 0; row < rows(); row++) {
        float row_max = std::numeric_limits<float>::lowest();
        float row_sum = 0.0f;
        for (int col = 0; col < cols(); col++) {
            row_max = (at(row, col) > row_max ? at(row, col) : row_max);
        }
        for (int col = 0; col < cols(); col++) {
            row_sum += std::exp(at(row, col) - row_max);
        }
        for (int col = 0; col < cols(); col++) {
            output.set(row, col, std::exp(at(row, col) - row_max) / row_sum);
        }
    }
    return output;
}

Tensor Tensor::matmul(const Tensor& t) const { /* C = A @ B */
    assert(cols() == t.rows());
    // output tensor is of shape: node_->rows, t.node_->cols
    Tensor output(node_->rows, t.node_->cols);

    for (int p = 0; p < node_->rows; p++) {
        for (int q = 0; q < t.node_->cols; q++) {
            float val = 0.0f;
            for (int k = 0; k < node_->cols; k++) {
                val += (at(p, k) * t.at(k, q));
            }
            output.set(p, q, val);
        }
    }

    // Save away for the backward pass
    auto a = node_;
    auto b = t.node_;
    /* Note: here we save the raw pointer otherwise we get a circular ref with
     * shared pointers */
    auto out = output.node_.get();

    output.node_->backward = [a, b, out]() {
        // compute local gradient
        Tensor dOut = Tensor(out->rows, out->cols, out->grad);
        Tensor dA = dOut.matmul(Tensor(*b).transpose()); /* dA = dOut @ B.T */
        Tensor dB = Tensor(*a).transpose().matmul(dOut); /* dB = A.T @ dOut */

        // flow the gradients backward
        for (int i = 0; i < dA.node_->rows; i++)
            for (int j = 0; j < dA.node_->cols; j++)
                a->grad[i * a->row_stride + j * a->col_stride] += dA.at(i, j);

        for (int i = 0; i < dB.node_->rows; i++)
            for (int j = 0; j < dB.node_->cols; j++)
                b->grad[i * b->row_stride + j * b->col_stride] += dB.at(i, j);
    };

    output.node_->prev.push_back(a);
    output.node_->prev.push_back(b);
    return output;
}

Tensor Tensor::clone() const {
    Tensor output(node_->rows, node_->cols, node_->data);
    return output;
}

/* Construct a topological sort of the graph, then walk through the sorted
 * nodes, calling the lambda for each node */
void Tensor::backward(void) {
    std::stack<Node*> stack;
    std::unordered_set<Node*> visited;

    // initialize gradient for the root node
    std::fill(node_->grad.begin(), node_->grad.end(), 1.0f);

    // Initialize root of toposort with the current node
    DFSVisit(node_.get(), visited, stack);

    while (!stack.empty()) {
        Node* top = stack.top();
        if (top->backward) top->backward();
        stack.pop();
    }
}

Tensor Tensor::transpose() const {
    Tensor t;
    t.node_ = std::make_shared<Node>(node_->cols, node_->rows, node_->col_stride, node_->row_stride,
                                     node_->data);
    return t;
}

void Tensor::adjust_weights(float lr) {
    assert(node_->data.size() == node_->grad.size());
    // Element-wise subtraction
    for (size_t i = 0; i < node_->data.size(); ++i) {
        node_->data[i] -= lr * node_->grad[i];
    }
}

std::vector<float> Tensor::flatten() const {
    std::vector<float> output;
    output.reserve(node_->rows * node_->cols);
    if (node_->row_stride == node_->cols && node_->col_stride == 1)
        output = node_->data;
    else {
        for (int i = 0; i < node_->rows; i++) {
            for (int j = 0; j < node_->cols; j++) {
                output.push_back(at(i, j));
            }
        }
    }
    return output;
}

void Tensor::DFSVisit(Tensor::Node* curr, std::unordered_set<Node*>& visited,
                      std::stack<Node*>& stack) {
    visited.insert(curr);
    for (const auto& p : curr->prev) {
        if (!visited.contains(p.get())) {
            DFSVisit(p.get(), visited, stack);
        }
    }
    stack.push(curr);
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << std::fixed << std::setprecision(5);
    for (int i = 0; i < t.node_->rows; ++i) {
        for (int j = 0; j < t.node_->cols; ++j) {
            os << std::setw(10) << t.at(i, j);
        }
        os << std::endl;
    }
    return os;
}

}  // namespace tn
