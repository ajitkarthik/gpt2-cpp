#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <span>
#include <string>
#include <system_error>

class MappedFile {
   public:
    explicit MappedFile(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd == -1)
            throw std::system_error(errno, std::generic_category(), "open:" + std::string(path));

        struct stat st{};
        if (::fstat(fd, &st) == -1) {
            int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat:" + std::string(path));
        }

        size_ = static_cast<std::size_t>(st.st_size);

        // mmap doesn't accept a zero-length mapping.
        if (size_ == 0) {
            ::close(fd);
            return;
        }

        void* mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        int error = errno;
        ::close(fd);  // The mapping remains valid after closing the fd.

        if (mapping == MAP_FAILED)
            throw std::system_error(error, std::generic_category(), "mmap" + std::string(path));

        data_ = static_cast<const std::byte*>(mapping);
    }

    ~MappedFile() {
        if (size_ != 0) ::munmap(const_cast<std::byte*>(data_), size_);
    }

    MappedFile(const MappedFile&) = delete;                                    // copy constructor
    MappedFile& operator=(const MappedFile&) = delete;                         // copy assignment
    MappedFile(MappedFile&& other) : data_(other.data_), size_(other.size_) {  // move constructor
        other.data_ = nullptr;
        other.size_ = 0;
    }

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

    // template <typename T>
    // [[nodiscard]] std::span<const T> span_at(std::size_t byte_offset, std::size_t count) const;
    // TODO: Endianness adjustment needs to be made here. Assuming this is run on Apple silicon or
    // x86 for now.
    template <typename T>
    [[nodiscard]] std::span<const T> span_at(std::size_t byte_offset, std::size_t count) const {
        const std::size_t bytes = count * sizeof(T);
        assert(byte_offset <= size_);
        assert(bytes <= (size_ - byte_offset));  // no overflow
        assert(byte_offset % alignof(T) == 0);   // safe to cast
        const std::byte* p = data_ + byte_offset;
        return {reinterpret_cast<const T*>(p), count};
    }

    std::int32_t to_int32(std::span<const std::byte> bytes);
    std::size_t size() { return size_; }

   private:
    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};