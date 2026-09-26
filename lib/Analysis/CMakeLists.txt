add_mlir_library(AsyncCheckAnalysis
  CheckAsyncCorrectness.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRNVGPUDialect
  MLIRPass
  MLIRSCFDialect
)
