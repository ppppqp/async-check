//===- CheckAsyncCorrectness.cpp - GPU correctness checker ------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsyncCheck/Analysis/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace async_check {
namespace {

class CheckAsyncCorrectnessPass
    : public mlir::PassWrapper<CheckAsyncCorrectnessPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CheckAsyncCorrectnessPass)

  llvm::StringRef getArgument() const final {
    return "check-async-correctness";
  }

  llvm::StringRef getDescription() const final {
    return "Check supported GPU synchronization and asynchronous-copy rules";
  }

  void runOnOperation() override {
    // Deliberately no verification claim yet. Analyses will be added one
    // diagnostic family at a time, beginning with divergent barriers.
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCheckAsyncCorrectnessPass() {
  return std::make_unique<CheckAsyncCorrectnessPass>();
}

void registerAsyncCheckPasses() {
  mlir::PassRegistration<CheckAsyncCorrectnessPass>();
}

} // namespace async_check
