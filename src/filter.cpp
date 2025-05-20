#include "image.hpp"
#include "util.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <oneapi/tbb/parallel_for_each.h>
#include <random>
#include <sched.h>
#include <span>
#include <fstream>

namespace filt {

int linear_channel::base_offset_elems() const {
  assert(base_offset_bytes % elem_width_bytes == 0);
  return base_offset_bytes / elem_width_bytes;
}

int linear_channel::stride_x_elems() const {
  assert(stride_x_bytes % elem_width_bytes == 0);
  return stride_x_bytes / elem_width_bytes;
}

int linear_channel::stride_y_elems() const {
  assert(stride_y_bytes % elem_width_bytes == 0);
  return stride_y_bytes / elem_width_bytes;
}

int linear_channel::offset_elems(int x, int y) const {
  return base_offset_elems()
    + x * stride_x_elems()
    + y * stride_y_elems();
}

[[maybe_unused]] static constexpr float approx_exp1(float x) {
  constexpr float a = (1 << 23) / 0.69314718f;
  constexpr float b = (1 << 23) * (127 - 0.043677448f);
  x = a * x + b;
  constexpr float c = (1 << 23);
  constexpr float d = (1 << 23) * 255;
  if (x < c) {
    x = 0.0f;
  } else if (x > d) {
    x = d;
  }
  return std::bit_cast<float>(static_cast<uint32_t>(x));
}

[[maybe_unused]] static constexpr float approx_exp_line(float x) {
  return std::max(0.f, 1.f + 0.3f * x);
}

constexpr int radius = 3;

using normal = std::array<float, 3>;

constexpr static float dot(const normal& a, const normal& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

struct csv_dumper: nonmovable {
  std::ofstream csv = std::ofstream("hist.csv");
  double probability = 1e-2;
  std::uniform_real_distribution<double> dist;
  std::mt19937_64 rng;

  explicit csv_dumper(const char* name, double prob):
    csv(name),
    probability(prob),
    dist(0., 1.),
    rng(std::random_device()())
  {}

  template<typename First, typename... Rest>
  void report(First&& first, Rest&&... rest) {
    if (dist(rng) < probability) {
      csv << std::forward<First>(first);
      ((csv << "," << std::forward<Rest>(rest)), ...);
      csv << '\n';
    }
  }
};

constexpr static std::pair<int, int> rotate_ij(int direction, int i, int j) {
  switch (direction) {
    case 0: return {i, j};   // down
    case 1: return {j, -i};  // left
    case 2: return {-i, -j}; // up
    case 3: return {-j, i};  // right
  }
  __builtin_unreachable();
}

static int shift_origin(int origin, int width, int dx, int dy) {
#ifdef BLOCKS
  int cell_x = origin & block_mask;
  int cell_y = (origin >> block_shift) & block_mask;
  int block_id = origin >> (2 * block_shift);

  cell_x += dx;
  cell_y += dy;

  if (cell_x < 0) {
    --block_id;
  } else if (cell_x >= block_size) {
    ++block_id;
  }

  if (cell_y < 0) {
    block_id -= width / block_size;
  } else if (cell_y >= block_size) {
    block_id += width / block_size;
  }

  cell_x &= block_mask;
  cell_y &= block_mask;
  return (block_id << (2 * block_shift)) | (cell_y << block_shift) | cell_x;
#else
  return origin + dy * width + dx;
#endif
}

void linear_filter(image_meta& meta, filter_streams s) {
  int total_pixels = meta.total_pixels();
  assert_release(std::ssize(s.dst) == total_pixels);
  assert_release(std::ssize(s.color) == total_pixels);
  assert_release(std::ssize(s.albedo) == total_pixels);
  assert_release(std::ssize(s.aux) == total_pixels);
  assert_release(std::ssize(s.interleaved_normals) == 3 * total_pixels);

  const int width = meta.width;

  float* __restrict z = s.aux.data();
  float* __restrict out = s.dst.data();
  const float* __restrict normals = s.interleaved_normals.data();

  auto get_normal = [&](int origin) -> normal {
    normal result;
    std::memcpy(
      result.data(),
      normals + 3 * origin,
      sizeof(result));
    return result;
  };

  for (int i = 0; i < total_pixels; ++i) {
    z[i] = s.color[i] / s.albedo[i];
  }

#ifdef BLOCKS
  int redzone = block_size * (width + 1);
  if (redzone % 16 != 0) {
    __builtin_unreachable();
  }
#else
  int redzone = radius * (width + 1);
#endif

  for (int origin = redzone; origin < total_pixels - redzone; ++origin) {
    float zorigin = z[origin];
    normal norigin = get_normal(origin);
    float value = zorigin;
    float weight = 1.f;

    _Pragma("unroll")
    for (int direction = 0; direction < 4; ++direction) {
      normal nprev = norigin;
      float ndotprev;

      _Pragma("unroll")
      for (int i = 1; i <= radius; ++i) {
        _Pragma("unroll")
        for (int j = -i; j < +i; ++j) {
          auto [dx, dy] = rotate_ij(direction, i, j);
          int offset = shift_origin(origin, width, dx, dy);;

          normal nhere = get_normal(offset);

          float ndot = dot(nprev, nhere);
          constexpr float threshold = 1.01f;
          if (ndot < 0.7f
          || (i > 1 && (ndot > ndotprev * threshold || ndotprev > ndot * threshold))) {
            goto quit_direction;
          }

          float gdist = std::exp((i*i + j*j) * (-1.f / (1 + 2 * radius)));

          float zhere = z[offset];
          float idiff = (zhere - zorigin);
          float gintensity = approx_exp1(idiff * idiff * (-1.f / 25.f));

          float factor = gdist * gintensity;
          value += zhere * factor;
          weight += factor;

          if (j == 0) {
            nprev = nhere;
            ndotprev = ndot;
          }
        }
      }

    quit_direction:;
    }

    float alb = s.albedo[origin];
    float final = alb * value / weight;
    out[origin] = final;
  }

  // for (int i = 0; i < total_pixels; ++i) {
  //   // s.aux2[i] = s.dst[i] / 10.f;
  //   s.dst[i] *= s.albedo[i];
  // }
}

}  // namespace filt