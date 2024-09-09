file(GLOB_RECURSE SRCS *.cu *.h *.hpp *.cpp)

if(NOT BUILD_SHARED_LIBS)
  cuda_add_library(bm_base ${SRCS})
else()
  cuda_add_library(bm_base SHARED ${SRCS})
endif()
target_link_libraries(bm_base benchmark::benchmark pthread bmlib)
