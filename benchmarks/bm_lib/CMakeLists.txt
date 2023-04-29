file(GLOB_RECURSE SRCS *.cu *.h *.hpp)

if(NOT BUILD_SHARED_LIBS)
  cuda_add_library(bm_lib ${SRCS})
else()
  cuda_add_library(bm_lib SHARED ${SRCS})
endif()
target_link_libraries(bm_lib benchmark::benchmark pthread)
