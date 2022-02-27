#include <CL/sycl.hpp>
#include <benchmark/benchmark.h>
#include <string>

namespace utils {
//-------------------------------------------------------------------------//
std::string get_sycl_implementation() {
#ifdef __LLVM_SYCL__
  std::string sycl_impl = "LLVM";
#else
  std::string sycl_impl = "hipSYCL";
#endif
  return sycl_impl;
}
//-------------------------------------------------------------------------//
auto create_queue() {
#ifdef TARGET_GPU
  cl::sycl::queue queue{cl::sycl::gpu_selector()};
#else
  cl::sycl::queue queue{cl::sycl::cpu_selector()};
#endif
  return queue;
}
//-------------------------------------------------------------------------//
/// Arguments simple parallel loop for
static void simple_arguments(benchmark::internal::Benchmark *b) {
  const std::int64_t max_size = 1024 * 1024 * 1024;
  const std::int64_t min_size = 1024;
  for (std::int64_t size = min_size; size <= max_size; size *= 2)
    b->Args({0, 0, size});
}
//-------------------------------------------------------------------------//
/// Arguments for axpby with nd range
static void axpby_arguments(benchmark::internal::Benchmark *b) {
#ifdef TARGET_GPU
  std::vector<std::int64_t> work_group_sizes = {32, 64, 128, 256, 512, 1024};
#else
  std::vector<std::int64_t> work_group_sizes = {16, 32, 64, 128};
#endif
  const std::int64_t max_size = 1024 * 1024 * 1024;
  const std::int64_t min_size = 1024;
  std::int64_t bs = 1; // batchsize
  for (std::int64_t ws : work_group_sizes)
    for (std::int64_t size = min_size; size <= max_size; size *= 2)
      b->Args({ws, bs, size});
}
//-------------------------------------------------------------------------//
/// Arguments for axpby with nd range
static void reduction_arguments(benchmark::internal::Benchmark *b) {
#ifdef TARGET_GPU
  std::vector<std::int64_t> work_group_sizes = {32, 64, 128, 256, 512, 1024};
#else
  std::vector<std::int64_t> work_group_sizes = {16, 32, 64, 128};
#endif
  std::vector<std::int64_t> batch_sizes = {1, 8, 16, 64};
  const std::int64_t max_size = 1024 * 1024 * 1024;
  const std::int64_t min_size = 1024;
  for (std::int64_t ws : work_group_sizes)
    for (std::int64_t bs : batch_sizes)
      for (std::int64_t size = min_size; size <= max_size; size *= 2)
        b->Args({ws, bs, size});
}
} // namespace utils