#include <CL/sycl.hpp>

using namespace cl;

namespace kernel {
// Kernel Implementation
//-----------------------------------------------//
template <class T> inline T divceil(T numerator, T denominator) {
  return (numerator / denominator + (numerator % denominator > 0));
}
//-------------------------------------------------------------------------//
/// Compute execution range for a 1d vector of size "length" with workgroups
/// of size wgs and batch size bs
sycl::nd_range<1> get_execution_range(std::size_t length, std::size_t wgs,
                                      std::size_t bs = 1) {
  std::size_t num_batches = divceil(length, bs);
  std::size_t padded_length = wgs * divceil(num_batches, wgs);
  sycl::range local{wgs};
  sycl::range global{padded_length};
  return sycl::nd_range<1>{global, local};
}
//-------------------------------------------------------------------------//
/// Intialize vector on device with size and a given value
template <typename T>
auto init_vector(sycl::queue queue, std::size_t n, T value) {

  T *x = cl::sycl::malloc_device<T>(n, queue);
  if (x == nullptr)
    throw std::runtime_error(" Unable to allocate memoy - SYCL error");

  // Fill vector in parallel
  queue.parallel_for(sycl::range{n}, [=](sycl::id<1> i) { x[i] = value; });
  queue.wait();

  return x;
}
//-------------------------------------------------------------------------//
template <typename T>
sycl::event axpy(sycl::queue q, std::size_t n, T alpha, T beta,
                 const T *__restrict x, T *__restrict y, std::size_t wgs = 512,
                 const std::vector<sycl::event> &events = {}) {

  sycl::nd_range range = get_execution_range(n, wgs, 1);

  auto e = q.submit([&](sycl::handler &h) {
    h.depends_on(events);
    h.parallel_for(range, [=](sycl::nd_item<1> it) {
      const std::size_t i = it.get_global_id(0);
      if (i < n)
        y[i] = alpha * x[i] + beta * y[i];
    }); // End of the kernel function
  });   // End of command group

  return e;
}
// //-------------------------------------------------------------------------//
// /// Computes the vector dot product.
// /// alpha += x[i] * y[i]
// /// Note: This function returns synchronously
// template <typename T>
// auto dot(sycl::queue &queue, std::size_t n, const T *x, const T *y,
//          std::size_t wgs = 512, std::size_t bs = 8,
//          const std::vector<sycl::event> &events = {}) {

//   sycl::buffer<T> sum{1};
//   queue.submit([&](sycl::handler &h) {
//     h.depends_on(events);
// #ifdef __LLVM_SYCL__
//     auto init_to_zero = sycl::property::reduction::initialize_to_identity{};
//     auto reductor =
//         sycl::reduction(sum, h, T{0.0}, std::plus<T>(), init_to_zero);
// #else
//     sycl::accessor sum_acc{sum, h, sycl::no_init};
//     auto reductor = sycl::reduction(sum_acc, sycl::plus<T>());
// #endif
//     h.parallel_for(sycl::range<1>{n}, reductor,
//                    [=](sycl::id<1> idx, auto &sum) {
//                      const std::size_t i = idx.get(0);
//                      sum += (x[i] * y[i]);
//                    }); // End of the kernel function
//   });                  // End of command group

//   sycl::host_accessor sum_host{sum};
//   return sum_host[0];
// }
// //-----------------------------------------------//
// /// Computes the vector dot product.
// /// alpha += x[i] * y[i]
// /// Note: This function returns synchronously
// template <typename T>
// auto cg_update(sycl::queue &queue, std::size_t n, T alpha, const T *p,
//                const T *y, T *x, T *r, std::size_t wgs, std::size_t bs,
//                const std::vector<sycl::event> &events = {}) {

//   sycl::range local{wgs};
//   sycl::range global{n};

//   sycl::buffer<T> sum{1};
//   queue.submit([&](sycl::handler &h) {
//     h.depends_on(events);
// #ifdef __LLVM_SYCL__
//     auto init = sycl::property::reduction::initialize_to_identity{};
//     auto reductor = sycl::reduction(sum, h, T{0.0}, std::plus<T>(), init);
// #else
//     sycl::accessor sum_acc{sum, h, sycl::no_init};
//     auto reductor = sycl::reduction(sum_acc, T{0}, sycl::plus<T>());
// #endif
//     h.parallel_for(sycl::nd_range{global, local}, reductor,
//                    [=](sycl::nd_item<1> it, auto &sum) {
//                      const std::size_t i = it.get_global_id(0);
//                      r[i] = -alpha * y[i] + r[i];
//                      x[i] = alpha * p[i] + x[i];
//                      sum += r[i] * r[i];
//                    }); // End of the kernel function
//   });                  // End of command group

//   sycl::host_accessor sum_host{sum};
//   return sum_host[0];
// }
// //-----------------------------------------------//

} // namespace kernel