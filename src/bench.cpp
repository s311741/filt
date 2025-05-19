#include "image.hpp"
#include <benchmark/benchmark.h>

static void filtering(benchmark::State& bm) {
  auto spectral = filt::image("./exr/bistro_cafe.exr");
  auto albedo = filt::image("./exr/bistro_cafe_albedo.exr");

  for (auto&& _: bm) {
    (void) filt::naive_filter(spectral, albedo);
  }
}
BENCHMARK(filtering);

BENCHMARK_MAIN();