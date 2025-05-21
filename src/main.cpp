#include "image.hpp"
#include "mempool.hpp"
#include "util.hpp"
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/ranges.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/task_group.h>
#include <sched.h>
#include <string_view>
#include <unistd.h>

[[maybe_unused]]
static void remove_non_rgb_channels(filt::image_meta& meta) {
   meta.channels.erase(
    std::remove_if(
      meta.channels.begin(), meta.channels.end(),
      [](const filt::linear_channel& channel) {
        if (channel.name.length() != 1) {
          return true;
        }
        switch (channel.name[0]) {
          case 'R': case 'G': case 'B': return false;
          default: return true;
        };
      }),
    meta.channels.end());
}

int main(int argc, char** argv) try {
  if (argc < 2) {
    throw std::runtime_error("No input image filename");
  }

  auto gbuf = filt::image(argv[1]);

#if 0
  {
    auto timer = interval_timer();
    auto result = filt::naive_filter(gbuf);
    auto dt = timer.elapsed();
    log_out(
      "{:.3f}us & {:.3f} Mp/s",
      dt.count(), gbuf.meta.total_pixels() / dt.count());
    result.dump_png_rgb("out/naive.png");
    return 0;
  }
#endif

  filt::linear_channel color_channels[3] = {
    gbuf.meta.find_channel("R"),
    gbuf.meta.find_channel("G"),
    gbuf.meta.find_channel("B"),
  };
  filt::linear_channel albedo_channels[3] = {
    gbuf.meta.find_channel("Albedo.R"),
    gbuf.meta.find_channel("Albedo.G"),
    gbuf.meta.find_channel("Albedo.B"),
  };
  filt::linear_channel normal_channels[3] = {
    gbuf.meta.find_channel("Ns.X"),
    gbuf.meta.find_channel("Ns.Y"),
    gbuf.meta.find_channel("Ns.Z"),
  };

  auto pool = filt::memory_pool();

  auto color_mem = pool.upload_channels_interleave(0, gbuf, color_channels);
  auto albedo_mem = pool.upload_channels_interleave(0, gbuf, albedo_channels);
  auto normal_mem = pool.upload_channels_interleave(128, gbuf, normal_channels);

  auto dst_mem = pool.allocate<float>(192, 3 * gbuf.meta.total_pixels());
  auto z_mem = pool.allocate<float>(0, 3 * gbuf.meta.total_pixels());

  auto result_image = filt::image(gbuf.meta);
  auto z_image = filt::image(gbuf.meta);
  auto zf_image = filt::image(gbuf.meta);

  for (int i = 0; i < 10; ++i) {
    auto timer = interval_timer();
    filt::linear_filter(gbuf.meta, filt::filter_streams{
      .dst = dst_mem,
      .color = color_mem,
      .albedo = albedo_mem,
      .normals = normal_mem,
      .z = z_mem,
    });
    timer.report(gbuf.meta.total_pixels());
  }

} catch (const std::exception& ex) {
  fmt::print(
    stderr, fg(fmt::terminal_color::red) | fmt::emphasis::bold,
    "Error: {}\n", ex.what());
  return 1;
}