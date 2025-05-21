#pragma once

#include "util.hpp"
#include <cassert>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <boost/container/small_vector.hpp>


namespace filt {

using boost::container::small_vector;

struct linear_channel {
  std::string name;

  int elem_width_bytes;
  int base_offset_bytes;
  int stride_x_bytes;
  int stride_y_bytes;

  int base_offset_elems() const {
    assert(base_offset_bytes % elem_width_bytes == 0);
    return base_offset_bytes / elem_width_bytes;
  }

  int stride_x_elems() const {
    assert(stride_x_bytes % elem_width_bytes == 0);
    return stride_x_bytes / elem_width_bytes;
  }

  int stride_y_elems() const {
    assert(stride_y_bytes % elem_width_bytes == 0);
    return stride_y_bytes / elem_width_bytes;
  }

  int offset_elems(int x, int y) const {
    return base_offset_elems()
        + x * stride_x_elems()
        + y * stride_y_elems();
  }
};

struct image_meta {
  int width;
  int height;
  small_vector<linear_channel, 16> channels;

  int total_pixels() const {
    return width * height;
  }

  int storage_size() const {
    return total_pixels() * channels.size();
  }

  int find_channel_idx(std::string_view name) const;
  auto& find_channel(this auto& self, std::string_view name) {
    return self.channels[self.find_channel_idx(name)];
  }
};

struct image: noncopyable {
  image_meta meta;
  std::vector<float> data;

  image(
    const char* exr_filename,
    const std::function<bool(std::string_view)> channel_filter
  );
  explicit image(const char* exr_filename);

  explicit image(image_meta m):
    meta(std::move(m)),
    data(meta.storage_size())
  {}

  float sample(const linear_channel& channel, int x, int y) const {
    return data[channel.offset_elems(x, y)];
  }

  float sample(int channel_idx, int x, int y) const {
    return sample(meta.channels[channel_idx], x, y);
  }

  void put_channel_data(const linear_channel& channel, std::span<const float> newdata);
  std::span<float> get_channel_data(const linear_channel& channel);

  void dump_pngs_prefix(std::string_view prefix) const;
  void dump_png_rgb(const char* path) const;

  std::vector<unsigned char> data_to_u8() const;
};

[[nodiscard]] image naive_filter(image& gbuffer);

struct filter_streams {
  std::span<float> dst;
  std::span<const float> color;
  std::span<const float> albedo;
  std::span<const float> interleaved_normals;
  std::span<float> aux;
  std::span<float> aux2;
};
void linear_filter(image_meta& meta, filter_streams streams);

}  // namespace filt