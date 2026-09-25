add_mlir_library(AsyncCheckAnalysis
  CheckAsyncCorrectness.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
)
