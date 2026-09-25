//===- async-check-opt.cpp - async-check command-line driver -----*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AsyncCheck/Analysis/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  mlir::registerAllPasses();
  async_check::registerAsyncCheckPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "async-check modular optimizer driver\n", registry));
}
