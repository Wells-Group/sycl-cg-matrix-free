#include <CL/sycl.hpp>

#include "src.hpp"
#include "utils.hpp"

#include <benchmark/benchmark.h>

constexpr double value = 1. / 3.;

using namespace cl;
using namespace std::chrono;

template <typename T> static void bench_axpy(benchmark::State &state) {

  cl::sycl::queue queue{sycl::default_selector()};

  const std::size_t work_group_size = state.range(0);
  const std::size_t global_size = state.range(1);

  T *x = kernel::init_vector<T>(queue, global_size, value);
  T *y = kernel::init_vector<T>(queue, global_size, value);
  T alpha = 1.;
  T beta = 0.5;

  for (auto _ : state) {
    auto start = high_resolution_clock::now();
    kernel::axpy(queue, global_size, alpha, beta, x, y, work_group_size).wait();
    auto end = high_resolution_clock::now();
    double t = duration_cast<duration<double>>(end - start).count();
    state.SetIterationTime(t);
  }
  cl::sycl::free(x, queue);
  cl::sycl::free(y, queue);

  int num_bytes = sizeof(T);
  state.SetItemsProcessed(3 * num_bytes * global_size * state.iterations());
  std::string impl = utils::get_sycl_implementation();
  state.SetLabel(impl);
}

// template <typename T> static void bench_dot(benchmark::State &state) {

//   const std::size_t work_group_size = state.range(0);
//   const std::size_t global_size = state.range(1);
//   cl::sycl::queue q{sycl::cpu_selector()};

//   T *x = kernel::init_vector<T>(q, global_size, value);
//   T *y = kernel::init_vector<T>(q, global_size, value);
//   // T* z = kernel::init_vector<T>(q, 1e8, value);

//   T result{0.};

//   for (auto _ : state) {
//     auto start = std::chrono::high_resolution_clock::now();
//     result = kernel::dot(q, global_size, x, y, work_group_size);
//     benchmark::DoNotOptimize(result);
//     q.wait();
//     auto end = std::chrono::high_resolution_clock::now();
//     auto t =
//         std::chrono::duration_cast<std::chrono::duration<double>>(end -
//         start);
//     state.SetIterationTime(t.count());
//   }
//   cl::sycl::free(x, q);
//   cl::sycl::free(y, q);

//   std::string impl = get_sycl_implementation();
//   int num_bytes = sizeof(T);
//   state.SetItemsProcessed(2 * num_bytes * global_size * state.iterations());
//   state.SetLabel(impl);
// }

// template <typename T> static void bench_cg_update(benchmark::State &state) {
//   const std::size_t work_group_size = state.range(0);
//   const std::size_t global_size = state.range(1);
//   std::cout << global_size;
//   cl::sycl::queue q{sycl::cpu_selector()};

//   T *x = kernel::init_vector<T>(q, global_size, value);
//   T *r = kernel::init_vector<T>(q, global_size, value);
//   T *p = kernel::init_vector<T>(q, global_size, value);
//   T *y = kernel::init_vector<T>(q, global_size, value);

//   T alpha{value};
//   T result{1.};

//   for (auto _ : state) {
//     q.fill<T>(x, result, global_size).wait();
//     auto start = std::chrono::steady_clock::now();
//     result +=
//         kernel::cg_update(q, global_size, alpha, p, y, x, r,
//         work_group_size);
//     benchmark::DoNotOptimize(result);
//     q.wait();
//     auto end = std::chrono::steady_clock::now();
//     auto t =
//         std::chrono::duration_cast<std::chrono::duration<double>>(end -
//         start);
//     state.SetIterationTime(t.count());
//   }

//   sycl::free(x, q);
//   sycl::free(r, q);
//   sycl::free(p, q);
//   sycl::free(y, q);

//   std::string impl = get_sycl_implementation();
//   int num_bytes = sizeof(T);
//   state.SetItemsProcessed(6 * num_bytes * global_size * state.iterations());
//   state.SetLabel(impl);
// }

/// Arguments for axpy
static void custom_arguments(benchmark::internal::Benchmark *b) {
  std::vector<std::int64_t> work_group_sizes = {16, 32, 64, 128};
  const std::int64_t max_size = 1024 * 1024 * 1024;
  const std::int64_t min_size = 1024;
  for (std::int64_t ws : work_group_sizes)
    for (std::int64_t size = min_size; size <= max_size; size *= 2)
      b->Args({ws, size});
}

// Register axpby benchmarks
BENCHMARK_TEMPLATE(bench_axpy, double)
    ->Apply(custom_arguments)
    ->UseManualTime()
    ->Iterations(10);

// // Register dot benchmarks
// BENCHMARK_TEMPLATE(bench_dot, double)
//     ->Apply(custom_arguments)
//     ->UseManualTime()
//     ->Iterations(300);

// // Register fused update benchmarks
// BENCHMARK_TEMPLATE(bench_cg_update, double)
//     ->Apply(custom_arguments)
//     ->UseManualTime()
//     ->Iterations(300);

// Run benchmarks
int main(int argc, char **argv) {
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  ::benchmark::RunSpecifiedBenchmarks();
}