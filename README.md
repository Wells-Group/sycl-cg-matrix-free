# SYCL benchmarks

## Building with hiPSYCL
Requires hipSYCL - https://github.com/illuhad/hipSYCL
### CPUs

```bash
export HIPSYCL_TARGETS=omp
mkdir build
cd build
cmake -DSYCL_IMPL=hipSYCL -DSYCL_TARGET=cpu ..
make 
```

### NVIDIA GPUS

```bash
export HIPSYCL_TARGETS=cuda:sm_XX
export CUDA_PATH=/path/to/cuda
mkdir build
cd build
cmake -DSYCL_IMPL=hipSYCL -DSYCL_TARGET=GPU ..
make
```
---
## Building with Intel LLVM
Requires Intell LLVM/SYCL fork - https://github.com/intel/llvm .

Instructions to install Intel's LLVM-Based SYCL can be found [here](https://intel.github.io/llvm-docs/).

### CPUs

```bash
export CXX=dpcpp
mkdir build_llvm
cd build_llvm
cmake -DSYCL_IMPL=LLVM -DSYCL_TARGET=CPU ..
make 
```

### NVIDIA GPUS

```bash
export CUDA_PATH=/path/to/cuda
export CXX=clang++
mkdir build
cd build
cmake -DSYCL_IMPL=LLVM -DSYCL_TARGET=GPU ..
make
```
---
## Running benchmarks
### Running all benchmarks
Output results on screen:
```bash
./sycl
```

Save benchmarks to file (eg: `output.txt`):
```bash
./sycl --benchmark_format=csv --benchmark_out={output.txt}
```

### Running a specific benchmark
Running axpby only:
```bash
./sycl --benchmark_filter=axpby --benchmark_format=csv --benchmark_out={output.txt}
```