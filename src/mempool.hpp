#pragma once
#include "image.hpp"
#include "util.hpp"
#include <span>
#include <stdexcept>

namespace filt {

struct memory_pool: nonmovable {
  std::span<std::byte> memory;
  ptrdiff_t top = 0;

  explicit memory_pool();
  ~memory_pool();

  void prefault_memory();

  template<typename T>
  [[nodiscard]] std::span<T> allocate(int offset_bytes, int size_elems) {
    assert_release(offset_bytes % sizeof(T) == 0);
    int size_bytes = size_elems * sizeof(T);
    int size_pages = (size_elems * sizeof(T) + offset_bytes + 4095) / 4096;

    size_bytes = size_pages * 4096;
    assert_release(offset_bytes <= size_bytes);

    if (top + size_bytes > std::ssize(memory)) {
      throw std::runtime_error("out of image filterer premapped memory");
    }

    void* ptr = memory.data() + top + offset_bytes;
    std::span<T> result(reinterpret_cast<T*>(ptr), size_elems);
    top += size_bytes;
    assert_release(top % 4096 == 0);
    return result;
  }

  [[nodiscard]] std::span<float> upload_channel(
    int offset,
    const image& image,
    const linear_channel& channel);

  [[nodiscard]] std::span<float> upload_channels_interleave(
    int offset,
    const image& image,
    std::span<const linear_channel> channels);
};

}  // namespace filt