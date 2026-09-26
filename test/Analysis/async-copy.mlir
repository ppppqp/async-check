// RUN: async-check-opt %s --check-async-correctness --verify-diagnostics

gpu.module @async_copy_tests {
  gpu.func @read_before_wait(
      %source: memref<16xf32>,
      %shared: memref<16xf32, #gpu.address_space<workgroup>>) kernel {
    %c0 = arith.constant 0 : index
    // expected-note@+1 {{overlapping asynchronous copy was issued here}}
    %copy = nvgpu.device_async_copy %source[%c0], %shared[%c0], 4
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %group = nvgpu.device_async_create_group %copy
    // expected-error@+1 {{async-copy destination is read before a sufficient wait}}
    %read = memref.load %shared[%c0] : memref<16xf32, #gpu.address_space<workgroup>>
    gpu.return
  }

  gpu.func @read_after_wait(
      %source: memref<16xf32>,
      %shared: memref<16xf32, #gpu.address_space<workgroup>>) kernel
      attributes {gpu.known_block_size = array<i32: 16, 1, 1>} {
    %tid = gpu.thread_id x
    %copy = nvgpu.device_async_copy %source[%tid], %shared[%tid], 1
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %group = nvgpu.device_async_create_group %copy
    nvgpu.device_async_wait %group
    %read = memref.load %shared[%tid] : memref<16xf32, #gpu.address_space<workgroup>>
    gpu.return
  }

  gpu.func @cross_thread_read_after_wait(
      %source: memref<16xf32>,
      %shared: memref<16xf32, #gpu.address_space<workgroup>>) kernel
      attributes {gpu.known_block_size = array<i32: 16, 1, 1>} {
    %tid = gpu.thread_id x
    %c1 = arith.constant 1 : index
    %neighbor = arith.addi %tid, %c1 : index
    // expected-note@+1 {{possibly overlapping shared-memory store is here}}
    %copy = nvgpu.device_async_copy %source[%tid], %shared[%tid], 1
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %group = nvgpu.device_async_create_group %copy
    nvgpu.device_async_wait %group
    // expected-warning@+1 {{possible cross-thread shared-memory RAW hazard; no recognized workgroup barrier separates this load from an earlier store}}
    %read = memref.load %shared[%neighbor] : memref<16xf32, #gpu.address_space<workgroup>>
    gpu.return
  }

  gpu.func @partial_wait_leaves_group_pending(
      %source: memref<16xf32>,
      %shared: memref<16xf32, #gpu.address_space<workgroup>>) kernel {
    %c0 = arith.constant 0 : index
    // expected-note@+1 {{overlapping asynchronous copy was issued here}}
    %copy = nvgpu.device_async_copy %source[%c0], %shared[%c0], 4
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %group = nvgpu.device_async_create_group %copy
    nvgpu.device_async_wait %group {numGroups = 1 : i32}
    // expected-error@+1 {{async-copy destination is read before a sufficient wait}}
    %read = memref.load %shared[%c0] : memref<16xf32, #gpu.address_space<workgroup>>
    gpu.return
  }

  gpu.func @disjoint_read(
      %source: memref<16xf32>,
      %shared: memref<16xf32, #gpu.address_space<workgroup>>) kernel {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %copy = nvgpu.device_async_copy %source[%c0], %shared[%c0], 4
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %group = nvgpu.device_async_create_group %copy
    %read = memref.load %shared[%c8] : memref<16xf32, #gpu.address_space<workgroup>>
    gpu.return
  }

  gpu.func @unknown_overlap(
      %source: memref<16xf32>,
      %shared: memref<16xf32, #gpu.address_space<workgroup>>,
      %copy_index: index, %read_index: index) kernel {
    %copy = nvgpu.device_async_copy %source[%copy_index], %shared[%copy_index], 1
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %group = nvgpu.device_async_create_group %copy
    // expected-remark@+1 {{async-check: unknown whether this read overlaps an incomplete asynchronous copy}}
    %read = memref.load %shared[%read_index] : memref<16xf32, #gpu.address_space<workgroup>>
    gpu.return
  }

  gpu.func @overlapping_copy(
      %source: memref<16xf32>,
      %shared: memref<16xf32, #gpu.address_space<workgroup>>) kernel {
    %c0 = arith.constant 0 : index
    // expected-note@+1 {{previous asynchronous copy was issued here}}
    %first = nvgpu.device_async_copy %source[%c0], %shared[%c0], 4
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    // expected-error@+1 {{async-copy destination overlaps a previous incomplete asynchronous copy}}
    %second = nvgpu.device_async_copy %source[%c0], %shared[%c0], 4
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %group = nvgpu.device_async_create_group %first, %second
    gpu.return
  }

  gpu.func @write_before_wait(
      %source: memref<16xf32>,
      %shared: memref<16xf32, #gpu.address_space<workgroup>>, %value: f32)
      kernel {
    %c0 = arith.constant 0 : index
    // expected-note@+1 {{overlapping asynchronous copy was issued here}}
    %copy = nvgpu.device_async_copy %source[%c0], %shared[%c0], 4
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %group = nvgpu.device_async_create_group %copy
    // expected-error@+1 {{shared-memory write overlaps an incomplete asynchronous copy}}
    memref.store %value, %shared[%c0] : memref<16xf32, #gpu.address_space<workgroup>>
    gpu.return
  }

  gpu.func @partial_wait_completes_oldest_group(
      %source: memref<16xf32>,
      %shared: memref<16xf32, #gpu.address_space<workgroup>>) kernel {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %first = nvgpu.device_async_copy %source[%c0], %shared[%c0], 4
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %first_group = nvgpu.device_async_create_group %first
    // expected-note@+1 {{overlapping asynchronous copy was issued here}}
    %second = nvgpu.device_async_copy %source[%c8], %shared[%c8], 4
      : memref<16xf32> to memref<16xf32, #gpu.address_space<workgroup>>
    %second_group = nvgpu.device_async_create_group %second
    nvgpu.device_async_wait %second_group {numGroups = 1 : i32}
    gpu.barrier
    %oldest_is_complete = memref.load %shared[%c0] : memref<16xf32, #gpu.address_space<workgroup>>
    // expected-error@+1 {{async-copy destination is read before a sufficient wait}}
    %newest_is_pending = memref.load %shared[%c8] : memref<16xf32, #gpu.address_space<workgroup>>
    gpu.return
  }
}
