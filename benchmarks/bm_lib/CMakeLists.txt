file(GLOB_RECURSE SRCS *.cc *.h *.hpp)

if(NOT BUILD_SHARED_LIBS)
  add_library(bm_lib ${SRCS})
else()
  add_library(bm_lib SHARED ${SRCS})
endif()
target_link_libraries(bm_lib benchmark::benchmark pthread)
