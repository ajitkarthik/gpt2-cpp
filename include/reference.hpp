// For file format, see the cpp file.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mappedfile.hpp"

class Reference {
   public:
    Reference(const std::string path);
    MappedFile mp;
    int32_t B;
    int32_t T;
    int32_t C;
    int32_t L;
    int32_t Vp;
    int32_t num_tensors;
    int32_t logit_rows;
    // Weights offsets
    size_t offset_encoded;
    std::vector<size_t> offset_residual;
    size_t offset_lnf;
    size_t offset_logits;
    // Weights sizes
    size_t size_encoded;
    std::vector<size_t> size_residual;
    size_t size_lnf;
    size_t size_logits;
    template <typename T>
    [[nodiscard]] std::span<const T> span_at(std::size_t byte_offset, std::size_t count) const;
    std::span<const float> spanAtIndex(const int index) const;
};