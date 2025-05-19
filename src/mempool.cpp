#include "mempool.hpp"
#include <fmt/ranges.h>
#include <ranges>
#include <sys/mman.h>

namespace filt {

memory_pool::memory_pool() {
  constexpr ptrdiff_t memory_size = 500 * 1024 * 1024;
  void* mapped = ::mmap(
    nullptr,
    memory_size,
    PROT_READ | PROT_WRITE,
    MAP_PRIVATE | MAP_ANONYMOUS,
    -1, 0);
  if (mapped == MAP_FAILED) {
    throw errno_error("mmap");
  }

  memory = std::span(reinterpret_cast<std::byte*>(mapped), memory_size);
}

void memory_pool::prefault_memory() {
  constexpr ptrdiff_t step = 4096;
  for (ptrdiff_t i = 0; i < std::ssize(memory); i += step) {
    *static_cast<volatile std::byte*>(&memory[i]) = {};
  }
}

memory_pool::~memory_pool() {
  ptrdiff_t total = memory.size();
  log_out(
    "Memory used: {} KiB / {} KiB ({}%)",
    top / 1024, total / 1024, 100 * top / total);
  ::munmap(memory.data(), memory.size_bytes());
}


static void assert_valid_channel(const image_meta& meta, const linear_channel& channel) {
  // otherwise the implementation of memory management
  // around channels would have to be different
  assert_release(channel.elem_width_bytes == sizeof(float));
  assert_release(channel.stride_x_elems() == 1);
  assert_release(channel.stride_y_elems() == meta.width);
}

std::span<float> memory_pool::upload_channel(
  int alloc_offset,
  const image& image,
  const linear_channel& channel
) {
  int total_pixels = image.meta.total_pixels();
  auto alloc = allocate<float>(alloc_offset, total_pixels);

  assert_valid_channel(image.meta, channel);
  std::uninitialized_move_n(
    image.data.data() + channel.base_offset_elems(),
    total_pixels,
    alloc.begin());

  log_out("Uploaded channel {} @ {}", channel.name, fmt::ptr(alloc.data()));
  return alloc;
}

std::span<float> memory_pool::upload_channels_interleave(
  int alloc_offset,
  const image& image,
  std::span<const linear_channel> channels
) {
  int channel_pixels = image.meta.total_pixels();
  int total_pixels = channel_pixels * std::ssize(channels);
  auto alloc = allocate<float>(alloc_offset, total_pixels);

  for (auto& channel: channels) {
    assert_valid_channel(image.meta, channel);
  }

  int offset = 0;
  for (int i = 0; i < channel_pixels; ++i) {
    for (auto& channel: channels) {
      alloc[offset++] = image.data[channel.base_offset_elems() + i];
    }
  }
  assert_release(offset == total_pixels);

  log_out("Interleaved {} channels @ {}: {}",
    std::ssize(channels),
    fmt::ptr(alloc.data()),
    fmt::join(std::views::transform(channels, &linear_channel::name), ", "));
  return alloc;
}

}  // namespace filt