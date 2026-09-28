#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mappedfile.hpp"
#include "tensor.hpp"

using tn::Tensor;

class Checkpoint {
   public:
    Checkpoint(const std::string path);
    Tensor& embed(std::vector<std::uint32_t> tokens);
    int32_t maxT;
    int32_t vocab;
    int32_t layers;
    int32_t nh;
    int32_t embsz;
    int32_t vocab_padded;
    MappedFile mp;
    // Weights offsets
    size_t offset_wte;
    size_t offset_wpe;
    size_t offset_ln1w;
    size_t offset_ln1b;
    size_t offset_qkvw;
    size_t offset_qkvb;
    size_t offset_attprojw;
    size_t offset_attprojb;
    size_t offset_ln2w;
    size_t offset_ln2b;
    size_t offset_fcw;
    size_t offset_fcb;
    size_t offset_fcprojw;
    size_t offset_fcprojb;
    size_t offset_lnfw;
    size_t offset_lnfb;
    // Weights sizes
    size_t size_wte;
    size_t size_wpe;
    size_t size_ln1w;
    size_t size_ln1b;
    size_t size_qkvw;
    size_t size_qkvb;
    size_t size_attprojw;
    size_t size_attprojb;
    size_t size_ln2w;
    size_t size_ln2b;
    size_t size_fcw;
    size_t size_fcb;
    size_t size_fcprojw;
    size_t size_fcprojb;
    size_t size_lnfw;
    size_t size_lnfb;
};