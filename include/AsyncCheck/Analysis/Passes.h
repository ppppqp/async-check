//===- Passes.h - async-check analysis passes -------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCCHECK_ANALYSIS_PASSES_H
#define ASYNCCHECK_ANALYSIS_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace async_check {

std::unique_ptr<mlir::Pass> createCheckAsyncCorrectnessPass();
void registerAsyncCheckPasses();

} // namespace async_check

#endif // ASYNCCHECK_ANALYSIS_PASSES_H
