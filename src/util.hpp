#pragma once
#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include <random>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <fstream>

void set_affinity(int from, int upto);

template<typename... Args>
auto fmt_runtime_error(fmt::format_string<Args...> format, Args&&... args) {
  return std::runtime_error(fmt::format(format, std::forward<Args>(args)...));
}

inline auto errno_error(const char* what) {
  return std::system_error(errno, std::generic_category(), what);
}

template<typename... Args>
void log_out(fmt::format_string<Args...> format, Args&&... args) {
#if 0
  fmt::println(stderr, format, std::forward<Args>(args)...);
#else
  (void) format;
  ((void) args, ...);
#endif
}

struct noncopyable {
  noncopyable() = default;
  noncopyable(const noncopyable&) = delete;
  noncopyable& operator=(const noncopyable&) = delete;
  noncopyable(noncopyable&&) = default;
  noncopyable& operator=(noncopyable&&) = default;
};

struct nonmovable {
  nonmovable() = default;
  nonmovable(const nonmovable&) = delete;
  nonmovable& operator=(const nonmovable&) = delete;
  nonmovable(nonmovable&&) = delete;
  nonmovable& operator=(nonmovable&&) = delete;
};

template<typename T>
std::span<T, 1> span1(T& t) {
  return std::span(&t);
}

#define assert_release(expr) \
  do { \
    if (!(expr)) [[unlikely]] { \
      fmt::println(stderr, "{}:{} assert: {}", __FILE__, __LINE__, #expr); \
      /*std::exit(1);*/ __builtin_trap(); \
    } \
  } while (false)


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
