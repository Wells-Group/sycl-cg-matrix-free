#include <string>

namespace utils {
std::string get_sycl_implementation() {
#ifdef __LLVM_SYCL__
  std::string sycl_impl = "LLVM";
#else
  std::string sycl_impl = "hipSYCL";
#endif
  return sycl_impl;
}
} // namespace utils