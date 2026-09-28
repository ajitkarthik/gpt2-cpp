#include "decoder.hpp"

#include "mappedfile.hpp"
#include "tensor.hpp"

using namespace decoder;
using namespace tn;
using namespace std;

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

// input shape (T, C), output shape (T, C)
Tensor Layer::forward(const Tensor& x) {
    Tensor out(x.rows(), x.cols());
    out = ln1_.forward(x);
    out = attn_.forward(out);
    out = out + x;  // add the residual stream
    out = ln2_.forward(out);
    out = ffn_.forward(out);
    out = out + x;  // add the residual stream
    return out;
}

Tensor FFN::forward(const Tensor& x) const {
    Tensor h = l1_.forward(x);
    gelu_(h);
    return l2_.forward(h);
}

Tensor Linear::forward(const Tensor& x) const {
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

Tensor LayerNorm::forward(const Tensor& x) const {
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

Tensor MHSA::forward(const Tensor& x) const {
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
                    sdp.set(sdp_row, sdp_col, -numeric_limits<float>::infinity());
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

Decoder::Decoder(Checkpoint& ckpt) {
    vocab_ = ckpt.vocab;                // vocabulary size
    nlayers_ = ckpt.layers;             // # of layers
    numheads_ = ckpt.nh;                // # of heads
    embedsize_ = ckpt.embsz;            // # embedding size
    vocab_padded_ = ckpt.vocab_padded;  // padded vocabulary size
    eps_ = 1.0e-5;                      // to prevent division by 0 in layernorm
    maxT_ = ckpt.maxT;                  // max context size

    layers_.reserve(nlayers_);
    // Load weights into decoder
    for (int i = 0; i < nlayers_; i++) {
        loadLayerWeights(
            i,
            ckpt.mp.span_at<float>(ckpt.offset_ln1w + (ckpt.size_ln1w / nlayers_) * i, embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_ln1b + (ckpt.size_ln1b / nlayers_) * i, embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_qkvw + (ckpt.size_qkvw / nlayers_) * i,
                                   3 * embedsize_ * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_qkvb + (ckpt.size_qkvb / nlayers_) * i,
                                   3 * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_attprojw + (ckpt.size_attprojw / nlayers_) * i,
                                   embedsize_ * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_attprojb + (ckpt.size_attprojb / nlayers_) * i,
                                   embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_ln2w + (ckpt.size_ln2w / nlayers_) * i, embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_ln2b + (ckpt.size_ln2b / nlayers_) * i, embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_fcw + (ckpt.size_fcw / nlayers_) * i,
                                   4 * embedsize_ * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_fcb + (ckpt.size_fcb / nlayers_) * i,
                                   4 * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_fcprojw + (ckpt.size_fcprojw / nlayers_) * i,
                                   embedsize_ * 4 * embedsize_),
            ckpt.mp.span_at<float>(ckpt.offset_fcprojb + (ckpt.size_fcprojb / nlayers_) * i,
                                   embedsize_));
    }

    // Load final layernorm weights
    loadLNWeights(ckpt.mp.span_at<float>(ckpt.offset_lnfw, embedsize_),
                  ckpt.mp.span_at<float>(ckpt.offset_lnfb, embedsize_));

    // Load vocabulary projection weights - these are the same as wte
    loadVocabProjWeights(ckpt.mp.span_at<float>(ckpt.offset_wte, embedsize_));

    // Initialize the token embedding matrix
    // wte (vocab_padded, embsz)
    wte_ =
        MatrixView(reinterpret_cast<const float*>(ckpt.mp.bytes().subspan(ckpt.offset_wte).data()),
                   vocab_padded_, embedsize_);

    // Initialize the positional encoding matrix
    // wpe (maxT, embsz)
    wpe_ =
        MatrixView(reinterpret_cast<const float*>(ckpt.mp.bytes().subspan(ckpt.offset_wpe).data()),
                   maxT_, embedsize_);
}

// Loads weights for a layer
void Decoder::loadLayerWeights(int i, span<const float> ln1w, span<const float> ln1b,
                               span<const float> qkvw, span<const float> qkvb,
                               span<const float> attprojw, span<const float> attprojb,
                               span<const float> ln2w, span<const float> ln2b,
                               span<const float> fcw, span<const float> fcb,
                               span<const float> fcprojw, span<const float> fcprojb) {
    layers_[i] = Layer(ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, ln2w, ln2b, fcw, fcb, fcprojw,
                       fcprojb, eps_, numheads_, embedsize_);
}

void Decoder::loadLNWeights(span<const float> lnfw, span<const float> lnfb) {
    finalLN_ = LayerNorm(lnfw, lnfb, eps_);
}

void Decoder::loadVocabProjWeights(span<const float> wte) {
    // GPT2 uses the same weights matrix for the final vocab projection as the token embedding
    // matrix
    vocabproj_ = Linear(MatrixView(wte.data(), vocab_padded_, embedsize_), nullopt);
}

// Looks up tokens in an embedding matrix -> (T, C)
// Adds positional embeddings -> (T, C)
// Returns a tensor of shape (T, C)
Tensor Decoder::embed(span<const int32_t> tokens) {
    int T = static_cast<int>(tokens.size());
    assert(T >= 0 && T <= wpe_.rows);
    Tensor x(T, wte_.cols);
    for (int i = 0; i < T; i++) {
        assert(tokens[i] >= 0 && tokens[i] < wte_.rows);
        for (int j = 0; j < wte_.cols; j++) {
            x.set(i, j, wte_.at(tokens[i], j) + wpe_.at(i, j));
        }
    }
    return x;
}

// The decoder expects x to be already embedded and positionally encoded
// The input into the decoder is of shape (T, C), and output are logits of shape (1, V)
// the predicted token
Tensor Decoder::forward(span<const int32_t> tokens) {
    Tensor x = embed(tokens);
    assert(x.cols() == embedsize_);  // shape of x (T, C)
    assert(x.rows() <= maxT_);       // make sure # of tokens <= max context window
    for (int i = 0; i < nlayers_; i++) {
        x = layers_[i].forward(x);
    }
    // Just grab the last position so we'll end up with (1, C)
    Tensor logits = Tensor(1, embedsize_);
    for (int i = 0; i < embedsize_; i++) {
        logits.set(0, i, x.at(x.rows() - 1, i));
    }

    logits = finalLN_.forward(logits);  // Final layerNorm, shape (1, C)
    assert(logits.rows() == 1 && logits.cols() == embedsize_);
    logits = vocabproj_.forward(logits);  // Vocabulary projection, shape (1, Vp)
    assert(logits.rows() == 1 && logits.cols() == vocab_padded_);
    // Remove the padded entries
    Tensor final_logits(1, vocab_);
    for (int i = 0; i < vocab_; i++) {
        final_logits.set(0, i, logits.at(0, i));
    }
    assert(final_logits.rows() == 1 && final_logits.cols() == vocab_);
    return final_logits;
}