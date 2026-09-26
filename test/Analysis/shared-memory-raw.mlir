// RUN: async-check-opt %s --check-async-correctness --verify-diagnostics

gpu.module @shared_memory_tests {
  gpu.func @missing_barrier(
      %shared: memref<32xf32, #gpu.address_space<workgroup>>, %value: f32)
      kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>} {
    %tid = gpu.thread_id x
    %c1 = arith.constant 1 : index
    %neighbor = arith.addi %tid, %c1 : index
    // expected-note@+1 {{possibly overlapping shared-memory store is here}}
    memref.store %value, %shared[%tid] : memref<32xf32, #gpu.address_space<workgroup>>
    // expected-warning@+1 {{possible cross-thread shared-memory RAW hazard; no recognized workgroup barrier separates this load from an earlier store}}
    %read = memref.load %shared[%neighbor] : memref<32xf32, #gpu.address_space<workgroup>>
    gpu.return
  }

  gpu.func @barrier_orders_access(
      %shared: memref<32xf32, #gpu.address_space<workgroup>>, %value: f32)
      kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>} {
    %tid = gpu.thread_id x
    %c1 = arith.constant 1 : index
    %neighbor = arith.addi %tid, %c1 : index
    memref.store %value, %shared[%tid] : memref<32xf32, #gpu.address_space<workgroup>>
    gpu.barrier
    %read = memref.load %shared[%neighbor] : memref<32xf32, #gpu.address_space<workgroup>>
    gpu.return
  }

  gpu.func @thread_private_index(
      %shared: memref<32xf32, #gpu.address_space<workgroup>>, %value: f32)
      kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>} {
    %tid = gpu.thread_id x
    memref.store %value, %shared[%tid] : memref<32xf32, #gpu.address_space<workgroup>>
    %read = memref.load %shared[%tid] : memref<32xf32, #gpu.address_space<workgroup>>
    gpu.return
  }
}
