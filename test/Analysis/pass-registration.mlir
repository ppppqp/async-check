// RUN: async-check-opt %s --check-async-correctness | FileCheck %s

// The initial pass shell must parse, run, and preserve input without claiming
// that any synchronization property has been verified.
module {
  // CHECK: func.func @smoke_test
  func.func @smoke_test() {
    return
  }
}
