#pragma once
#include <sstream>

// Helper template to safely convert strings to any type
template <typename T>
inline bool convert_arg(const std::string& input, T& output) {
    std::stringstream ss(input);
    ss >> output;
    return !ss.fail() && ss.eof();  // Return true if conversion entirely succeeded
}