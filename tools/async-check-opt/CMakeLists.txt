set(LIBS
  AsyncCheckAnalysis
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
)

add_llvm_executable(async-check-opt async-check-opt.cpp)
llvm_update_compile_flags(async-check-opt)
target_link_libraries(async-check-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(async-check-opt)
