#include "mappedfile.hpp"
using namespace std;

int32_t MappedFile::to_int32(std::span<const std::byte> bytes) {
    uint32_t u =
        static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
        (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
    return static_cast<int32_t>(u);
}