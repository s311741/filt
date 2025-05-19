#include "util.hpp"
#include <sched.h>

void set_affinity(int from, int upto) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int cpu = from; cpu < upto; ++cpu) {
    CPU_SET(cpu, &cpuset);
  }
  if (sched_setaffinity(gettid(), sizeof(cpuset), &cpuset) == -1) {
    throw errno_error("setaffinity");
  }
}