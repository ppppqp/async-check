// RUN: async-check-opt %s --check-async-correctness --verify-diagnostics

gpu.module @barrier_tests {
  gpu.func @divergent_barrier() kernel
      attributes {gpu.known_block_size = array<i32: 32, 1, 1>} {
    %tid = gpu.thread_id x
    %c16 = arith.constant 16 : index
    // expected-note@+1 {{this control predicate varies across the known block dimensions}}
    %predicate = arith.cmpi ult, %tid, %c16 : index
    scf.if %predicate {
      // expected-error@+1 {{workgroup barrier is control-divergent; not every thread reaches the same barrier instance}}
      gpu.barrier
    }
    gpu.return
  }

  // A thread-dependent expression can still be uniform over the known block.
  gpu.func @uniform_barrier() kernel
      attributes {gpu.known_block_size = array<i32: 32, 1, 1>} {
    %tid = gpu.thread_id x
    %c32 = arith.constant 32 : index
    %predicate = arith.cmpi ult, %tid, %c32 : index
    scf.if %predicate {
      gpu.barrier
    }
    gpu.return
  }

  gpu.func @unknown_barrier_uniformity(
      %predicate: i1,
      %shared: memref<32xf32, #gpu.address_space<workgroup>>, %value: f32) kernel
      attributes {gpu.known_block_size = array<i32: 32, 1, 1>} {
    %tid = gpu.thread_id x
    %c1 = arith.constant 1 : index
    %neighbor = arith.addi %tid, %c1 : index
    scf.if %predicate {
      // expected-note@+1 {{possibly overlapping shared-memory store is here}}
      memref.store %value, %shared[%tid] : memref<32xf32, #gpu.address_space<workgroup>>
      // expected-remark@+1 {{async-check: unknown whether every workgroup thread reaches this barrier}}
      gpu.barrier
      // The unknown barrier is not allowed to discharge visibility facts.
      // expected-warning@+1 {{possible cross-thread shared-memory RAW hazard; no recognized workgroup barrier separates this load from an earlier store}}
      %read = memref.load %shared[%neighbor] : memref<32xf32, #gpu.address_space<workgroup>>
    }
    gpu.return
  }
}
