// #include <CL/sycl.hpp>

// #include "src.hpp"
// #include "utils.hpp"

// #include <benchmark/benchmark.h>

// using namespace cl;
// using namespace std::chrono;

// constexpr std::size_t N = 10;
// constexpr double value{1. / 3.};

// //-------------------------------------------------------------------------//
// template <typename T> static void bench_axpby(benchmark::State &state) {
//   sycl::queue queue = utils::create_queue();
//   [[maybe_unused]] std::size_t wgs = state.range(0);
//   [[maybe_unused]] std::size_t bs = state.range(1);
//   [[maybe_unused]] std::size_t size = state.range(2);

//   T *x = kernel::init_vector<T>(queue, size, value);
//   T *y = kernel::init_vector<T>(queue, size, value);
//   T alpha = 1.;
//   T beta = 0.5;

//   for (auto _ : state) {
//     auto start = high_resolution_clock::now();
//     kernel::axpby(queue, size, alpha, beta, x, y, wgs).wait();
//     auto end = high_resolution_clock::now();
//     double t = duration_cast<duration<double>>(end - start).count();
//     state.SetIterationTime(t);
//   }
//   cl::sycl::free(x, queue);
//   cl::sycl::free(y, queue);

//   int num_bytes = sizeof(T);
//   state.SetItemsProcessed(3 * num_bytes * size * state.iterations());
//   std::string impl = utils::get_sycl_implementation();
//   state.SetLabel(impl);
// }
// //-------------------------------------------------------------------------//
// template <typename T> static void bench_axpby_simple(benchmark::State &state)
// {
//   sycl::queue queue = utils::create_queue();
//   [[maybe_unused]] std::size_t wgs = state.range(0);
//   [[maybe_unused]] std::size_t bs = state.range(1);
//   [[maybe_unused]] std::size_t size = state.range(2);

//   T *x = kernel::init_vector<T>(queue, size, value);
//   T *y = kernel::init_vector<T>(queue, size, value);
//   T alpha = 1.;
//   T beta = 0.5;

//   for (auto _ : state) {
//     auto start = high_resolution_clock::now();
//     kernel::axpby(queue, size, alpha, beta, x, y).wait();
//     auto end = high_resolution_clock::now();
//     double t = duration_cast<duration<double>>(end - start).count();
//     state.SetIterationTime(t);
//   }
//   cl::sycl::free(x, queue);
//   cl::sycl::free(y, queue);

//   int num_bytes = sizeof(T);
//   state.SetItemsProcessed(3 * num_bytes * size * state.iterations());
//   std::string impl = utils::get_sycl_implementation();
//   state.SetLabel(impl);
// }
// //-------------------------------------------------------------------------//
// template <typename T> static void bench_dot(benchmark::State &state) {
//   sycl::queue queue = utils::create_queue();
//   [[maybe_unused]] std::size_t wgs = state.range(0);
//   [[maybe_unused]] std::size_t bs = state.range(1);
//   [[maybe_unused]] std::size_t size = state.range(2);

//   T *x = kernel::init_vector<T>(queue, size, value);
//   T *y = kernel::init_vector<T>(queue, size, value);

//   T result{0.};

//   for (auto _ : state) {
//     auto start = high_resolution_clock::now();
//     result = kernel::dot(queue, size, x, y, wgs, bs);
//     benchmark::DoNotOptimize(result);
//     auto end = high_resolution_clock::now();
//     double t = duration_cast<duration<double>>(end - start).count();
//     state.SetIterationTime(t);
//   }
//   cl::sycl::free(x, queue);
//   cl::sycl::free(y, queue);

//   std::string impl = utils::get_sycl_implementation();
//   int num_bytes = sizeof(T);
//   state.SetItemsProcessed(2 * num_bytes * size * state.iterations());
//   state.SetLabel(impl);
// }
// //-------------------------------------------------------------------------//
// template <typename T> static void bench_dot_simple(benchmark::State &state) {
//   sycl::queue queue = utils::create_queue();
//   [[maybe_unused]] std::size_t wgs = state.range(0);
//   [[maybe_unused]] std::size_t bs = state.range(1);
//   [[maybe_unused]] std::size_t size = state.range(2);

//   T *x = kernel::init_vector<T>(queue, size, value);
//   T *y = kernel::init_vector<T>(queue, size, value);

//   T result{0.};

//   for (auto _ : state) {
//     auto start = high_resolution_clock::now();
//     result = kernel::dot(queue, size, x, y);
//     benchmark::DoNotOptimize(result);
//     auto end = high_resolution_clock::now();
//     double t = duration_cast<duration<double>>(end - start).count();
//     state.SetIterationTime(t);
//   }
//   cl::sycl::free(x, queue);
//   cl::sycl::free(y, queue);

//   std::string impl = utils::get_sycl_implementation();
//   int num_bytes = sizeof(T);
//   state.SetItemsProcessed(2 * num_bytes * size * state.iterations());
//   state.SetLabel(impl);
// }
// //-------------------------------------------------------------------------//
// template <typename T>
// static void bench_update_unfused(benchmark::State &state) {
//   sycl::queue queue = utils::create_queue();
//   [[maybe_unused]] std::size_t wgs = state.range(0);
//   [[maybe_unused]] std::size_t bs = state.range(1);
//   [[maybe_unused]] std::size_t size = state.range(2);

//   T *x = kernel::init_vector<T>(queue, size, value);
//   T *r = kernel::init_vector<T>(queue, size, value);
//   T *p = kernel::init_vector<T>(queue, size, value);
//   T *y = kernel::init_vector<T>(queue, size, value);

//   T alpha{value};
//   T result{1.};

//   for (auto _ : state) {
//     auto start = high_resolution_clock::now();
//     auto e0 = kernel::axpby(queue, size, -alpha, T{1.}, y, r, wgs);
//     auto e1 = kernel::axpby(queue, size, alpha, T{1.}, p, x, wgs);
//     result = kernel::dot(queue, size, r, r, {e0, e1});
//     benchmark::DoNotOptimize(result);
//     auto end = high_resolution_clock::now();
//     double t = duration_cast<duration<double>>(end - start).count();
//     state.SetIterationTime(t);
//   }

//   sycl::free(x, queue);
//   sycl::free(r, queue);
//   sycl::free(p, queue);
//   sycl::free(y, queue);

//   std::string impl = utils::get_sycl_implementation();
//   int num_bytes = sizeof(T);
//   state.SetItemsProcessed(7 * num_bytes * size * state.iterations());
//   state.SetLabel(impl);
// }
// //
// -------------------------------------------------------------------------//
// template <typename T> static void bench_cg_update(benchmark::State &state) {
//   sycl::queue queue = utils::create_queue();
//   [[maybe_unused]] std::size_t wgs = state.range(0);
//   [[maybe_unused]] std::size_t bs = state.range(1);
//   [[maybe_unused]] std::size_t size = state.range(2);

//   T *x = kernel::init_vector<T>(queue, size, value);
//   T *r = kernel::init_vector<T>(queue, size, value);
//   T *p = kernel::init_vector<T>(queue, size, value);
//   T *y = kernel::init_vector<T>(queue, size, value);

//   T alpha{value};
//   T result{1.};

//   for (auto _ : state) {
//     auto start = high_resolution_clock::now();
//     result += kernel::cg_update(queue, size, alpha, p, y, x, r, wgs, bs);
//     benchmark::DoNotOptimize(result);
//     auto end = high_resolution_clock::now();
//     double t = duration_cast<duration<double>>(end - start).count();
//     state.SetIterationTime(t);
//   }

//   sycl::free(x, queue);
//   sycl::free(r, queue);
//   sycl::free(p, queue);
//   sycl::free(y, queue);

//   std::string impl = utils::get_sycl_implementation();
//   int num_bytes = sizeof(T);
//   state.SetItemsProcessed(6 * num_bytes * size * state.iterations());
//   state.SetLabel(impl);
// }
// //
// -------------------------------------------------------------------------//
// template <typename T>
// static void bench_cg_update_simple(benchmark::State &state) {
//   sycl::queue queue = utils::create_queue();
//   [[maybe_unused]] std::size_t wgs = state.range(0);
//   [[maybe_unused]] std::size_t bs = state.range(1);
//   [[maybe_unused]] std::size_t size = state.range(2);

//   T *x = kernel::init_vector<T>(queue, size, value);
//   T *r = kernel::init_vector<T>(queue, size, value);
//   T *p = kernel::init_vector<T>(queue, size, value);
//   T *y = kernel::init_vector<T>(queue, size, value);

//   T alpha{value};
//   T result{1.};

//   for (auto _ : state) {
//     auto start = high_resolution_clock::now();
//     result += kernel::cg_update(queue, size, alpha, p, y, x, r);
//     benchmark::DoNotOptimize(result);
//     auto end = high_resolution_clock::now();
//     double t = duration_cast<duration<double>>(end - start).count();
//     state.SetIterationTime(t);
//   }

//   sycl::free(x, queue);
//   sycl::free(r, queue);
//   sycl::free(p, queue);
//   sycl::free(y, queue);

//   std::string impl = utils::get_sycl_implementation();
//   int num_bytes = sizeof(T);
//   state.SetItemsProcessed(6 * num_bytes * size * state.iterations());
//   state.SetLabel(impl);
// }
// //
// -------------------------------------------------------------------------//
// // Register benchmarks
// //
// -------------------------------------------------------------------------//
// // Register nd range axpby benchmarks
// BENCHMARK_TEMPLATE(bench_axpby, double)
//     ->Apply(utils::axpby_arguments)
//     ->UseManualTime()
//     ->Iterations(N);

// // Register axpby simple benchmarks
// BENCHMARK_TEMPLATE(bench_axpby_simple, double)
//     ->Apply(utils::simple_arguments)
//     ->UseManualTime()
//     ->Iterations(N);

// // Register axpby simple benchmarks
// BENCHMARK_TEMPLATE(bench_dot, double)
//     ->Apply(utils::reduction_arguments)
//     ->UseManualTime()
//     ->Iterations(N);

// // Register axpby simple benchmarks
// BENCHMARK_TEMPLATE(bench_dot_simple, double)
//     ->Apply(utils::simple_arguments)
//     ->UseManualTime()
//     ->Iterations(N);

// // Register update unfused benchmarks
// BENCHMARK_TEMPLATE(bench_update_unfused, double)
//     ->Apply(utils::reduction_arguments)
//     ->UseManualTime()
//     ->Iterations(N);

// // Register fused update benchmarks
// BENCHMARK_TEMPLATE(bench_cg_update, double)
//     ->Apply(utils::reduction_arguments)
//     ->UseManualTime()
//     ->Iterations(N);

// // Register fused update benchmarks
// BENCHMARK_TEMPLATE(bench_cg_update_simple, double)
//     ->Apply(utils::simple_arguments)
//     ->UseManualTime()
//     ->Iterations(N);

// // Run benchmarks
// int main(int argc, char **argv) {
// #ifdef TARGET_GPU
//   std::cout << "targeting gpus";
// #endif
//   ::benchmark::Initialize(&argc, argv);
//   if (::benchmark::ReportUnrecognizedArguments(argc, argv))
//     return 1;
//   ::benchmark::RunSpecifiedBenchmarks();
// }

#include "src.hpp"
#include "utils.hpp"
#include <CL/sycl.hpp>

using T = double;

using namespace cl;
using namespace std::chrono;

int main(int argc, char **argv) {
  sycl::queue queue = utils::create_queue();
  std::size_t size = 1e5;
  T value = 2.;
  T *x = kernel::init_vector<T>(queue, size, value);
  T *y = kernel::init_vector<T>(queue, size, value);

  auto start = high_resolution_clock::now();
  T result = kernel::dot_group(queue, size, x, y, 512, 16);
  auto end = high_resolution_clock::now();
  double t = duration_cast<duration<double>>(end - start).count();
  std::cout << t << std::endl;
  std::cout << result << std::endl;


  return 0;
}