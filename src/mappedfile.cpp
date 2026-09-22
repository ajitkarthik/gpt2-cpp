#include "mappedfile.hpp"
using namespace std;

int32_t MappedFile::to_int32(std::span<const std::byte> bytes) {
    uint32_t u =
        static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
        (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
    return static_cast<int32_t>(u);
}

[[nodiscard]] std::span<const std::byte> MappedFile::bytes() const noexcept {
    return {data_, size_};
}

[[nodiscard]] std::span<const float> MappedFile::floats_at(std::size_t byte_offset,
                                                           std::size_t count) const {
    const std::size_t bytes = count * sizeof(float);
    assert(byte_offset <= size_);
    assert(bytes <= (size_ - byte_offset));     // no overflow
    assert(byte_offset % alignof(float) == 0);  // safe to cast
    const std::byte* p = data_ + byte_offset;
    return {reinterpret_cast<const float*>(p), count};
}