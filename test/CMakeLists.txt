set(ASYNC_CHECK_TEST_DEPENDS
  async-check-opt
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-async-check "Running async-check regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ASYNC_CHECK_TEST_DEPENDS}
)
set_target_properties(check-async-check PROPERTIES FOLDER "Tests")

add_lit_testsuites(ASYNC_CHECK ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${ASYNC_CHECK_TEST_DEPENDS}
)
