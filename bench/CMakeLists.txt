# ROS-free benchmark harness: builds the planner cores straight from src/,
# without a colcon workspace. Built in the `gpu` container:
#   cmake -S bench -B bench/build -G Ninja && cmake --build bench/build
cmake_minimum_required(VERSION 3.24)
project(pathfinder_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

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
