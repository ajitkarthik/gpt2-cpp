#include "loader.hpp"
using namespace std;

std::int32_t MappedFile::to_int32(std::span<const std::byte> bytes) {
    return static_cast<std::int32_t>(bytes[0]) | (static_cast<std::int32_t>(bytes[1]) << 8) |
           (static_cast<std::int32_t>(bytes[2]) << 16) |
           (static_cast<std::int32_t>(bytes[3]) << 24);
}