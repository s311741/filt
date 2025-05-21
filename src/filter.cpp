#include "image.hpp"
#include "util.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <oneapi/tbb/parallel_for_each.h>
#include <sched.h>
#include <span>

#if 1
#define unroll _Pragma("unroll")
#else
#define unroll
#endif

namespace filt {

constexpr int radius = 3;
using float3 = std::array<float, 3>;

constexpr static float dot(const float3& a, const float3& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

constexpr static std::pair<int, int> rotate_ij(int direction, int i, int j) {
  switch (direction) {
    case 0: return {i, j};   // down
    case 1: return {j, -i};  // left
    case 2: return {-i, -j}; // up
    case 3: return {-j, i};  // right
  }
  __builtin_unreachable();
}

image naive_filter(image& gbuf) {
  image_meta meta;
  meta.width = gbuf.meta.width;
  meta.height = gbuf.meta.height;
  const int stride_x = sizeof(float);
  const int stride_y = stride_x * meta.width;
  for (int i = 0; i < 3; ++i) {
    const char name[2] = {"RGB"[i], '\0'};
    meta.channels.push_back(linear_channel{
      .name = name,
      .elem_width_bytes = sizeof(float),
      .base_offset_bytes = i * int(sizeof(float)) * meta.total_pixels(),
      .stride_x_bytes = stride_x,
      .stride_y_bytes = stride_y,
    });

    auto color = gbuf.get_channel_data(gbuf.meta.find_channel(name));
    auto albedo = gbuf.get_channel_data(
      gbuf.meta.find_channel(fmt::format("Albedo.{}", name)));
    for (int j = 0; j < std::ssize(color); ++j) {
      color[j] /= albedo[j];
    }
  }
  image result(std::move(meta));

  const linear_channel& albr = gbuf.meta.find_channel("Albedo.R");
  const linear_channel& albg = gbuf.meta.find_channel("Albedo.G");
  const linear_channel& albb = gbuf.meta.find_channel("Albedo.B");
  const auto get_albedo = [&](int x, int y) {
    return float3{ gbuf.sample(albr, x, y),
                   gbuf.sample(albg, x, y),
                   gbuf.sample(albb, x, y) };
  };

  const linear_channel& nx = gbuf.meta.find_channel("Ns.X");
  const linear_channel& ny = gbuf.meta.find_channel("Ns.Y");
  const linear_channel& nz = gbuf.meta.find_channel("Ns.Z");
  const auto get_normal = [&](int x, int y) {
    return float3{ gbuf.sample(nx, x, y),
                   gbuf.sample(ny, x, y),
                   gbuf.sample(nz, x, y) };
  };

  const linear_channel& r = gbuf.meta.find_channel("R");
  const linear_channel& g = gbuf.meta.find_channel("G");
  const linear_channel& b = gbuf.meta.find_channel("B");
  const auto get_z = [&](int x, int y) {
    return float3{ gbuf.sample(r, x, y),
                   gbuf.sample(g, x, y),
                   gbuf.sample(b, x, y) };
  };

  for (int y = radius; y < meta.height - radius; ++y) {
    for (int x = radius; x < meta.width - radius; ++x) {
      float3 zorigin = get_z(x, y);
      float3 norigin = get_normal(x, y);
      float3 value = zorigin;
      float3 weight {1.f, 1.f, 1.f};

      unroll for (int direction = 0; direction < 4; ++direction) {
        float3 nprev = norigin;
        float ndotprev;
        unroll for (int i = 1; i <= radius; ++i) {
          unroll for (int j = -i; j < i; ++j) {
            auto [dx, dy] = rotate_ij(direction, i, j);
            int xx = x + dx;
            int yy = y + dy;
            float3 nhere = get_normal(xx, yy);
            float ndot = dot(nprev, nhere);
            constexpr float threshold = 1.01f;
            if (ndot < 0.7f
             || (i > 1 && (ndot > ndotprev * threshold
                        || ndotprev > ndot * threshold))) {
              goto kill_direction;
            }

            float gdist = std::exp(
              (i*i + j*j) * (-1.f / (1 + 2 + radius)));

            float3 zhere = get_z(xx, yy);
            unroll for (int k = 0; k < 3; ++k) {
              float id = zhere[i] - zorigin[i];
              float gintensity = std::exp(id * id * (-1.f / 25.f));
              float factor = gdist * gintensity;
              value[k] += zhere[k] * factor;
              weight[k] += factor;
            }

            if (j == 0) {
              nprev = nhere;
              ndotprev = ndot;
            }
          }
        }
      kill_direction:;
      }

      float3 alb = get_albedo(x, y);
      _Pragma("unroll") for (int i = 0; i < 3; ++i) {
        float final = alb[i] * value[i] / weight[i];
        result.data[result.meta.channels[i].offset_elems(x, y)] = final;
      }
    }
  }

  return result;
}

// =====================================================================

static constexpr float approx_exp1(float x) {
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

void linear_filter(image_meta& meta, filter_streams s) {
  const int total_pixels = meta.total_pixels();
  assert_release(std::ssize(s.dst) == 3 * total_pixels);
  assert_release(std::ssize(s.color) == 3 * total_pixels);
  assert_release(std::ssize(s.albedo) == 3 * total_pixels);
  assert_release(std::ssize(s.z) == 3 * total_pixels);
  assert_release(std::ssize(s.normals) == 3 * total_pixels);

  const int width = meta.width;

#define RESTRICT __restrict
// #define RESTRICT
  float* RESTRICT z = s.z.data();
  float* RESTRICT out = s.dst.data();
  const float* RESTRICT normals = s.normals.data();
  const float* RESTRICT albedo = s.albedo.data();

  auto get_z = [&](int at) -> float3 {
    float3 result;
    std::memcpy(result.data(), z + 3 * at, sizeof(result));
    return result;
  };

  auto get_albedo = [&](int at) -> float3 {
    float3 result;
    std::memcpy(result.data(), albedo + 3 * at, sizeof(result));
    return result;
  };

  auto get_normal = [&](int at) -> float3 {
    float3 result;
    std::memcpy(result.data(), normals + 3 * at, sizeof(result));
    return result;
  };

  for (int i = 0; i < 3 * total_pixels; ++i) {
    z[i] = s.color[i] / albedo[i];
  }

  const int redzone = radius * (width + 1);
  for (int origin = redzone; origin < total_pixels - redzone; ++origin) {
    float3 zorigin = get_z(origin);
    float3 norigin = get_normal(origin);
    float3 value = zorigin;
    float3 weight {1.f, 1.f, 1.f};

    unroll for (int direction = 0; direction < 4; ++direction) {
      float3 nprev = norigin;
      float ndotprev;

      unroll for (int i = 1; i <= radius; ++i) {
        unroll for (int j = -i; j < +i; ++j) {
          auto [dx, dy] = rotate_ij(direction, i, j);
          int offset = origin + dy * width + dx;

          float3 nhere = get_normal(offset);

          float ndot = dot(nprev, nhere);
          constexpr float threshold = 1.01f;
          if (ndot < 0.7f
          || (i > 1 && (ndot > ndotprev * threshold || ndotprev > ndot * threshold))) {
            goto kill_direction;
          }

          float gdist = std::exp((i*i + j*j) * (-1.f / (1 + 2 * radius)));

          float3 zhere = get_z(offset);

          unroll for (int k = 0; k < 3; ++k) {
            float id = (zhere[k] - zorigin[k]);
            float gintensity = approx_exp1(id * id * (-1.f / 25.f));
            float factor = gdist * gintensity;
            value[k] += zhere[k] * factor;
            weight[k] += factor;
          }

          if (j == 0) {
            nprev = nhere;
            ndotprev = ndot;
          }
        }
      }

    kill_direction:;
    }

    float3 alb = get_albedo(origin);
    for (int i = 0; i < 3; ++i) {
      float final = alb[i] * value[i] / weight[i];
      out[3 * origin + i] = final;
    }
  }
}

}  // namespace filt