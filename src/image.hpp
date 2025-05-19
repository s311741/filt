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

  int base_offset_elems() const;
  int stride_x_elems() const;
  int stride_y_elems() const;
  int offset_elems(int x, int y) const;
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

  void put_channel_data(const linear_channel& channel, std::span<const float> newdata);

  void dump_pngs(std::string_view dir) const;
  void dump_png_rgb(const char* path) const;

  std::vector<unsigned char> data_to_u8() const;
};

[[nodiscard]] image naive_filter(const image& spectral, const image& gbuffer);

struct filter_streams {
  std::span<float> dst;
  std::span<const float> color;
  std::span<const float> albedo;
  std::span<const float> interleaved_normals;
  std::span<float> aux;
  std::span<float> aux2;
};
void real_filter(image_meta& meta, filter_streams streams);

}  // namespace filt