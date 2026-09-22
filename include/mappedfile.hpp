#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <span>
#include <system_error>

class MappedFile {
   public:
    explicit MappedFile(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd == -1) throw std::system_error(errno, std::generic_category(), "open");

        struct stat st{};
        if (::fstat(fd, &st) == -1) {
            int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat");
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

        if (mapping == MAP_FAILED) throw std::system_error(error, std::generic_category(), "mmap");

        data_ = static_cast<const std::byte*>(mapping);
    }

    ~MappedFile() {
        if (size_ != 0) ::munmap(const_cast<std::byte*>(data_), size_);
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

    [[nodiscard]] std::span<const float> floats_at(std::size_t byte_offset,
                                                   std::size_t count) const;

    std::int32_t to_int32(std::span<const std::byte> bytes);

   private:
    const std::byte* data_ = nullptr;
    std::size_t size_ = 0;
};