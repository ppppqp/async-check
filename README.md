# async-check

`async-check` is an out-of-tree MLIR project for statically checking a focused
set of GPU synchronization mistakes:

- divergent block barriers;
- possible cross-thread shared-memory read-after-write hazards; and
- asynchronous-copy destinations consumed before a sufficient wait.

The first asynchronous-copy target is the NVGPU
`nvgpu.device_async_copy` operation family. TMA and `mbarrier` support is a
planned extension, not part of the initial guarantee. 

## Build

Use an LLVM/MLIR build from the revision recorded in
[`llvm-project.commit`](llvm-project.commit):

```sh
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build
cmake --build build --target check-async-check
```

Run the driver:

```sh
build/bin/async-check-opt input.mlir --check-async-correctness
```

## Layout

```text
include/AsyncCheck/   public C++ interfaces
lib/Analysis/        checker implementation
tools/async-check-opt opt-style command-line driver
test/Analysis/       CPU-only lit tests
test/Unsupported/    explicit proof-boundary tests
benchmarks/          future hardware corroboration corpus and runner
```
