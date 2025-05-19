#pragma once
#include <chrono>
#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include <sched.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

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
#if 1
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

struct interval_timer {
  using clock = std::chrono::steady_clock;
  clock::time_point time_started = clock::now();
  int cpu_started = sched_getcpu();
  std::string_view name;

  explicit interval_timer(std::string_view n = ""): name(n) {}

  void report() const {
    using dmicroseconds = std::chrono::duration<double, std::micro>;

    clock::time_point now = clock::now();
    int cpu = sched_getcpu();
    auto elapsed = std::chrono::duration_cast<dmicroseconds>(now - time_started);
    fmt::print(
      stderr, fg(fmt::terminal_color::bright_green),
      "Timer '{}': time {:.3f}us, cpu {}->{}\n",
      name, elapsed.count(), cpu_started, cpu);
  }
};

template<typename T>
std::span<T, 1> span1(T& t) {
  return std::span(&t);
}

#define assert_release(expr) \
  do { \
    if (!(expr)) [[unlikely]] { \
      fmt::println(stderr, "{}:{} assert: {}", __FILE__, __LINE__, #expr); \
      std::exit(1); \
    } \
  } while (false)
