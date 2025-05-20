#include "image.hpp"
#include "util.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <numeric>
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

static void touch(const float* ptr) {
  *static_cast<const volatile float*>(ptr);
}

[[gnu::noinline]]
static void naive_filter_channel(
  const image_meta& meta,
  const linear_channel& channel,
  std::span<float> dst,
  std::span<const float> src,
  std::span<const float> gbuffer
) {
  (void) gbuffer;
  constexpr int radius = 1;

  for (int y = radius; y + radius < meta.height; ++y) {
    for (int x = radius; x + radius < meta.width; ++x) {
      int off = channel.offset_elems(x, y);
      touch(&src[off]);
      touch(&dst[off]);
      // touch(&gbuffer[off]);
    }
  }
}

image naive_filter(const image& spectral, const image& gbuffer) {
  assert(spectral.meta.height == gbuffer.meta.height
      && spectral.meta.width == gbuffer.meta.width);

  image result(spectral.meta);

  // tbb::parallel_for_each
  std::for_each
  (
    spectral.meta.channels.begin(), spectral.meta.channels.end(),
    [&](const linear_channel& channel) {
      auto timer = interval_timer("filtering");
      naive_filter_channel(
        result.meta, channel, result.data,
        spectral.data, gbuffer.data);
      timer.report();
    }
  );

  return result;
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
constexpr int side = 1 + 2 * radius;

struct kernel {
  float values[side][side] = {};

  constexpr float sample(int dx, int dy) const {
    return values[dx + radius][dy + radius];
  }

  constexpr static kernel gaussian(float factor) {
    kernel result;
    float total = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        float g = approx_exp1((dx * dx + dy * dy) * factor);
        result.values[dx + radius][dy + radius] = g;
        total += g;
      }
    }
    for (int i = 0; i < side; ++i) {
      for (int j = 0; j < side; ++j) {
        result.values[i][j] /= total;
      }
    }
    return result;
  }
};

using normal = std::array<float, 3>;

#if 0
constexpr static float normaldiff(const normal& a, const normal& b) {
  // float x = a[1] * b[2] - a[2] * b[1];
  // float y = a[2] * b[0] - a[0] * b[2];
  // float z = a[0] * b[1] - a[1] * b[0];
  // return x * x + y * y + z * z;
  float x = a[0] - b[0];
  float y = a[1] - b[1];
  float z = a[2] - b[2];
  return std::abs(x) + std::abs(y) + std::abs(z);
}
#endif

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
    case 0: return {i, j};
    case 1: return {j, -i};
    case 2: return {-i, -j};
    case 3: return {-j, i};
  }
  __builtin_unreachable();
}

void real_filter(image_meta& meta, filter_streams s) {
  int total_pixels = meta.total_pixels();
  assert_release(std::ssize(s.dst) == total_pixels);
  assert_release(std::ssize(s.color) == total_pixels);
  assert_release(std::ssize(s.albedo) == total_pixels);
  assert_release(std::ssize(s.aux) == total_pixels);
  assert_release(std::ssize(s.aux2) == total_pixels);
  assert_release(std::ssize(s.interleaved_normals) == 3 * total_pixels);

  const int width = meta.width;

  auto get_normal = [&](int origin) -> normal {
    normal result;
    std::memcpy(
      result.data(),
      s.interleaved_normals.data() + 3 * origin,
      sizeof(result));
    return result;
  };

  auto z = s.aux;
  for (int i = 0; i < total_pixels; ++i) {
    z[i] = s.color[i] / s.albedo[i];
  }

  int redzone = radius * (width + 1);
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
          int offset = origin + dy * width + dx;
          normal nhere = get_normal(offset);

          float ndot = dot(nprev, nhere);
          constexpr float dn_threshold = 1.01f;
          if (ndot < 0.5f
          || (i > 1 && (ndot > ndotprev * dn_threshold || ndotprev > ndot * dn_threshold))) {
            s.aux2[origin] += 0.25f / i;
            goto quit_direction;
          }

          float dist2 = i*i + j*j;
          float gdist = std::exp(dist2 * (-1.f / (1 + 2 * radius)));

          float zhere = z[offset];
          float idiff = (zhere - zorigin);
          float gintensity = approx_exp1(idiff * idiff * (-1.f / 10.f));

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

    s.dst[origin] = value / weight;
  }

  for (int i = 0; i < total_pixels; ++i) {
    s.dst[i] *= s.albedo[i];
  }
}

}  // namespace filt