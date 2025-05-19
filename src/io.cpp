#include "image.hpp"
#include "util.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/base.h>
#include <fmt/ranges.h>
#include <ImathVec.h>
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>
#include <iterator>
#include <oneapi/tbb/parallel_for_each.h>
#include <png.h>
#include <sched.h>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tbb/parallel_for_each.h>
#include <vector>

class png_writer {
  png_structp write_struct = nullptr;
  png_infop info_struct = nullptr;
  FILE* out_stream = nullptr;

  void cleanup() {
    // libpng cleanup is messy, so do it all here
    png_destroy_write_struct(&write_struct, &info_struct);
    if (out_stream) {
      fclose(out_stream);
      out_stream = nullptr;
    }
  }

public:
  explicit png_writer(const char* filename) {
    try {
      write_struct = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
      if (!write_struct) {
        throw std::runtime_error("cannot create libpng write struct");
      }
      info_struct = png_create_info_struct(write_struct);
      if (!info_struct) {
        throw std::runtime_error("cannot create libpng info struct");
      }
      out_stream = fopen(filename, "wb");
      if (!out_stream) {
        throw errno_error("open png file");
      }
    } catch (...) {
      cleanup();
      throw;
    }

    png_init_io(write_struct, out_stream);
  }

  ~png_writer() {
    cleanup();
  }

  png_writer(png_writer&&) = delete;
  png_writer(const png_writer&) = delete;
  png_writer& operator=(png_writer&&) = delete;
  png_writer& operator=(const png_writer&) = delete;

  void write(int width, std::span<const unsigned char* const> rows, int color_type) && {
    int height = std::ssize(rows);
    png_set_IHDR(
      write_struct,
      info_struct,
      width, height, 8,
      color_type,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT);
    png_write_info(write_struct, info_struct);
    png_write_image(write_struct, const_cast<unsigned char**>(rows.data()));
    png_write_end(write_struct, nullptr);
  }

  void write_grayscale(int width, std::span<const unsigned char* const> rows) && {
    return std::move(*this).write(width, rows, PNG_COLOR_TYPE_GRAY);
  }

  void write_rgb_interleaved(int width, std::span<const unsigned char* const> rows) && {
    return std::move(*this).write(width, rows, PNG_COLOR_TYPE_RGB);
  }
};

namespace filt {

static image_meta meta_from_exr(
  Imf::InputFile& imf_image,
  const std::function<bool(std::string_view)> channel_filter
) {
  const auto imf_window = imf_image.header().dataWindow();
  const auto imf_size = imf_window.size() + Imath::V2i(1, 1);
  auto& imf_channels = imf_image.header().channels();

  image_meta meta;
  meta.width = imf_size.x;
  meta.height = imf_size.y;

  const int stride_x_bytes = sizeof(float);
  const int stride_y_bytes = meta.width * stride_x_bytes;
  int current_base_offset = 0;

  for (auto c = imf_channels.begin(); c != imf_channels.end(); ++c) {
    std::string name = c.name();

    if (!channel_filter(name)) {
      continue;
    }

    if (c.channel().type != Imf::PixelType::FLOAT) {
      throw fmt_runtime_error(
        "Channel {} in image {} is not single-precision float type",
        name,
        imf_image.fileName());
    }

    meta.channels.push_back(linear_channel{
      .name = std::move(name),
      .elem_width_bytes = sizeof(float),
      .base_offset_bytes = current_base_offset,
      .stride_x_bytes = stride_x_bytes,
      .stride_y_bytes = stride_y_bytes,
    });
    current_base_offset += sizeof(float) * meta.width * meta.height;
  }

  return meta;
}

int image_meta::find_channel_idx(std::string_view name) const {
  auto it = std::ranges::find(channels, name, &linear_channel::name);
  if (it == channels.end()) {
    throw fmt_runtime_error("channel {} not found", name);
  }
  return it - channels.begin();
}


image::image(
  const char* exr_filename,
  const std::function<bool(std::string_view)> channel_filter
) {
  auto exr = Imf::InputFile(exr_filename);
  meta = meta_from_exr(exr, channel_filter);

  if (meta.channels.empty()) {
    throw std::runtime_error("No spectral channels in image");
  }

  data.resize(meta.storage_size());

  Imf::FrameBuffer framebuffer;
  for (const linear_channel& channel: meta.channels) {
    framebuffer.insert(channel.name.c_str(), Imf::Slice(
      Imf::FLOAT,
      reinterpret_cast<char*>(data.data() + channel.base_offset_elems()),
      channel.stride_x_bytes,
      channel.stride_y_bytes));
  }

  exr.setFrameBuffer(framebuffer);
  exr.readPixels(0, meta.height-1);
  log_out("Done reading image {}", exr_filename);
}

image::image(const char* exr_filename):
  image(exr_filename, [](std::string_view) { return true; })
{}

void image::put_channel_data(const linear_channel& channel, std::span<const float> newdata) {
  assert_release(channel.stride_x_elems() == 1);
  assert_release(channel.stride_y_elems() == meta.width);
  assert_release(std::ssize(newdata) == meta.total_pixels());
  std::copy(newdata.begin(), newdata.end(), data.begin() + channel.base_offset_elems());
}

static uint8_t clamp_float_value(float f) {
  return std::clamp(f, 0.f, 1.0f) * 255.f;
}

void image::dump_pngs(std::string_view dir) const {
  std::vector<unsigned char> all_data(data.size());
  for (int i = 0; i < std::ssize(data); ++i) {
    all_data[i] = clamp_float_value(data[i]);
  }

  tbb::parallel_for_each(
    meta.channels.begin(), meta.channels.end(),
    [&](const linear_channel& channel) {
      std::vector<const unsigned char*> rows(meta.height);
      assert_release(channel.stride_x_elems() == 1); // else gathering all_data would have to work differently
      for (int i = 0; i < meta.height; ++i) {
        rows[i] = reinterpret_cast<const unsigned char*>(all_data.data())
          + channel.base_offset_elems()
          + i * channel.stride_y_elems();
      }
      auto filename = fmt::format("{}/{}.png", dir, channel.name);
      png_writer(filename.c_str()).write_grayscale(meta.width, rows);
      log_out("Done writing gray image {} on cpu {}", filename, sched_getcpu());
    });
}

void image::dump_png_rgb(const char* path) const {
  std::vector<unsigned char> all_data(3 * meta.total_pixels());
  linear_channel channels[3] = {
    meta.find_channel("R"),
    meta.find_channel("G"),
    meta.find_channel("B"),
  };
  for (auto& channel: channels) {
    assert_release(channel.stride_x_elems() == 1);
  }

  // interleave
  int offset = 0;
  for (int i = 0; i < meta.total_pixels(); ++i) {
    for (auto& channel: channels) {
      all_data[offset++] = clamp_float_value(data[channel.base_offset_elems() + i]);
    }
  }
  assert_release(offset == std::ssize(all_data));

  std::vector<const unsigned char*> rows(meta.height);
  for (int i = 0; i < meta.height; ++i) {
    rows[i] = reinterpret_cast<const unsigned char*>(all_data.data())
      + 3 * i * meta.width;
  }
  png_writer(path).write_rgb_interleaved(meta.width, rows);
  log_out("Done writing rgb image {} on cpu {}", path, sched_getcpu());
}

}  // namespace filt