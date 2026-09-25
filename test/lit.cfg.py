import os

import lit.formats
from lit.llvm import llvm_config

config.name = "ASYNC_CHECK"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.async_check_obj_root, "test")

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.add_tool_substitutions(
    ["FileCheck", "async-check-opt"],
    [config.async_check_tools_dir, config.llvm_tools_dir],
)
