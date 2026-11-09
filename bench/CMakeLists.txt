# ROS-free benchmark harness: builds the planner cores straight from src/ plus
# the CUDA planners, without a colcon workspace. Built in the `gpu` container:
#   cmake -S bench -B bench/build -G Ninja && cmake --build bench/build
cmake_minimum_required(VERSION 3.24)
project(pathfinder_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

option(PF_WITH_CUDA "Build the CUDA planners (if a CUDA compiler is found)" ON)
if(PF_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "No CUDA compiler found: building the CPU planners only")
    set(PF_WITH_CUDA OFF)
  endif()
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(OCTOMAP REQUIRED octomap)

set(PF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_library(pf_cores STATIC
  ${PF_SRC}/infra/pathfinder_core/src/voxel_grid.cpp
  ${PF_SRC}/algorithms/pathfinder_algo_astar/src/astar_core.cpp
  ${PF_SRC}/algorithms/pathfinder_algo_dijkstra/src/dijkstra_core.cpp
  ${PF_SRC}/algorithms/pathfinder_algo_dstar_lite/src/dstar_lite_core.cpp
  ${PF_SRC}/algorithms/pathfinder_algo_rrt/src/rrt_core.cpp
)
target_include_directories(pf_cores PUBLIC
  ${PF_SRC}/infra/pathfinder_core/include
  ${PF_SRC}/algorithms/pathfinder_algo_astar/include
  ${PF_SRC}/algorithms/pathfinder_algo_dijkstra/include
  ${PF_SRC}/algorithms/pathfinder_algo_dstar_lite/include
  ${PF_SRC}/algorithms/pathfinder_algo_rrt/include
  ${OCTOMAP_INCLUDE_DIRS}
)
target_link_directories(pf_cores PUBLIC ${OCTOMAP_LIBRARY_DIRS})
target_link_libraries(pf_cores PUBLIC ${OCTOMAP_LIBRARIES})

add_executable(pf_export_bt src/export_bt.cpp)
target_link_libraries(pf_export_bt PRIVATE pf_cores)

add_executable(pf_bench src/bench_main.cpp)
target_link_libraries(pf_bench PRIVATE pf_cores)

if(PF_WITH_CUDA)
  enable_language(CUDA)
  # 120 = Blackwell consumer (RTX 50xx); PTX for 89 keeps older GPUs working via JIT.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "89-virtual;120-real")
  endif()
  add_library(pf_gpu STATIC gpu/gpu_planners.cu)
  target_include_directories(pf_gpu PUBLIC gpu)
  set_target_properties(pf_gpu PROPERTIES CUDA_STANDARD 17)
  target_compile_options(pf_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
  target_link_libraries(pf_bench PRIVATE pf_gpu)
  target_compile_definitions(pf_bench PRIVATE PF_WITH_CUDA)
endif()
