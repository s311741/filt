#include "image.hpp"
#include "util.hpp"
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
  x = 1.f + 0.3f * x;
  if (x < 0.f) {
    x = 0.f;
  }
  return x;
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
    float value = 0;
    float weight = 0;

    float zorigin = z[origin];
    normal norigin = get_normal(origin);

    _Pragma("unroll") for (int dy = -radius; dy <= radius; ++dy) {
      _Pragma("unroll") for (int dx = -radius; dx <= radius; ++dx) {
        int offset = origin + dy * width + dx;
        float zhere = z[offset];
        normal nhere = get_normal(offset);

        float gspace = std::exp((dx * dx + dy * dy) * (-1.f / (1 + 2 * radius)));

        float idiff = (zhere - zorigin) / zorigin;
        float gintensity = approx_exp_line(idiff * idiff * (-1.f / 0.4f));

        float ndot = dot(nhere, norigin);
        float gnormal = ndot * ndot;
        if (ndot < 0.9995) {
          gnormal = 0.f;
        }

        float factor = gspace * gintensity * gnormal;
        value += zhere * factor;
        weight += factor;
      }
    }

    if (weight == 0.f) [[unlikely]] {
      s.dst[origin] = z[origin];
    } else {
      s.dst[origin] = value / weight;
    }
  }

  float sum_start = std::accumulate(z.begin() + redzone, z.end() - redzone, 0.f);
  float sum_end = std::accumulate(s.dst.begin() + redzone, s.dst.end() - redzone, 0.f);
  log_out("{} -> {}", sum_start, sum_end);

  for (int i = 0; i < total_pixels; ++i) {
    s.aux2[i] = s.dst[i] / 10.f;
    s.dst[i] *= s.albedo[i];
  }
}

}  // namespace filt