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

struct perf_scope: nonmovable {
  int child_pid;

  perf_scope() {
    int my_pid = getpid();
    child_pid = fork();
    if (child_pid == -1) {
      throw errno_error("fork");
    }

    if (child_pid == 0) {
      constexpr std::string_view events[] = {
        "L1-dcache-load-misses",
        "L1-dcache-loads",
        "LLC-load-misses",
        "LLC-loads",
      };
      auto cmdline = fmt::format("exec perf stat -p {} -e {}", my_pid, fmt::join(events, ","));
      execl("/bin/sh", "sh", "-c", cmdline.c_str(), nullptr);
      throw errno_error("execl");
    }

    setpgid(child_pid, 0);
  }

  ~perf_scope() {
    kill(-child_pid, SIGINT);
  }
};

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

  auto pool = filt::memory_pool();
  auto gbuf = filt::image(argv[1]);

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

  std::span<float> color_mem[3];
  std::span<float> albedo_mem[3];
  for (int i = 0; i < 3; ++i) {
    color_mem[i] = pool.upload_channel(0, gbuf, color_channels[i]);
    albedo_mem[i] = pool.upload_channel(64, gbuf, albedo_channels[i]);
  }
  auto normal_mem = pool.upload_channels_interleave(128, gbuf, normal_channels);

  auto dst_mem = pool.allocate<float>(192, gbuf.meta.total_pixels());
  auto aux_mem = pool.allocate<float>(0, gbuf.meta.total_pixels());
  auto aux2_mem = pool.allocate<float>(0, gbuf.meta.total_pixels());

  auto result_image = filt::image(gbuf.meta);
  auto z_image = filt::image(gbuf.meta);
  auto zf_image = filt::image(gbuf.meta);

  for (int ci = 0; ci < 3; ++ci) {
    std::ranges::fill(aux2_mem, 0.f);

    auto timer = interval_timer("filtering");
    filt::linear_filter(gbuf.meta, filt::filter_streams {
      .dst = dst_mem,
      .color = color_mem[ci],
      .albedo = albedo_mem[ci],
      .interleaved_normals = normal_mem,
      .aux = aux_mem,
      .aux2 = aux2_mem,
    });
    auto dt = timer.elapsed();
    log_out(
      "{:.3f}us - {:.3f} Mp/s",
      dt.count(), gbuf.meta.total_pixels() / dt.count());

    char target_name[2] = { "RGB"[ci], 0 };
    result_image.put_channel_data(
      result_image.meta.find_channel(target_name),
      dst_mem);
    zf_image.put_channel_data(
      zf_image.meta.find_channel(target_name),
      aux2_mem);
    // drops_image.put_channel_data(
    //   drops_image.meta.find_channel(target_name),
    //   aux2_mem);
    for (float& f: aux_mem) { f /= 10.f; }
    z_image.put_channel_data(z_image.meta.find_channel(target_name), aux_mem);
  }

  std::function<void()> tasks[] = {
    [&] {
      gbuf.unpack_all_channels();
      gbuf.dump_png_rgb("out/in.png");
    },
    [&] {
      result_image.unpack_all_channels();
      result_image.dump_png_rgb("out/out.png");
    },
    [&] {
      z_image.unpack_all_channels();
      z_image.dump_png_rgb("out/z.png");
    },
    [&] {
      zf_image.unpack_all_channels();
      zf_image.dump_png_rgb("out/zf.png");
    },
    // [&] {
    //   remove_non_rgb_channels(drops_image.meta);
    //   drops_image.dump_pngs_prefix("out/drops-");
    // },
  };
  tbb::parallel_for_each(tasks, [](auto& task) {task();});

} catch (const std::exception& ex) {
  fmt::print(
    stderr, fg(fmt::terminal_color::red) | fmt::emphasis::bold,
    "Error: {}\n", ex.what());
  return 1;
}