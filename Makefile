COMPOSE ?= docker compose
DEV := $(COMPOSE) run --rm -T dev
DEV_TTY := $(COMPOSE) run --rm --service-ports dev

TOOLCHAIN := /workspace/build/conan_toolchain.cmake
SANITIZERS_INCLUDE := /workspace/cmake/sanitizers.cmake

# Sanitizer runtime suppressions, applied to every in-container run. A normal
# Release binary has no sanitizer runtime to read these, so setting them
# unconditionally is a no-op there; for asan/tsan builds they silence the known
# third-party FastDDS / ROS / Gazebo noise documented in sanitizers/*.supp.
# asan.supp is LSan syntax, so it goes via LSAN_OPTIONS (not ASAN_OPTIONS).
SAN_RUNTIME := TSAN_OPTIONS=suppressions=/workspace/sanitizers/tsan.supp LSAN_OPTIONS=suppressions=/workspace/sanitizers/asan.supp

# Override with HEADLESS=true to run Gazebo server-only (no GUI). Useful when
# DISPLAY isn't set or the host doesn't have X11.
HEADLESS ?= false

# Sanitizer selection. Leave empty for a normal Release build. Accepted values:
#   asan, tsan, ubsan, asan+ubsan, tsan+ubsan
# When set, the build switches to Debug for better stack traces.
SANITIZER ?=

ifeq ($(SANITIZER),)
  SAN_CMAKE_ARGS :=
  BUILD_TYPE := Release
else ifeq ($(SANITIZER),asan)
  SAN_CMAKE_ARGS := -DENABLE_ASAN=ON
  BUILD_TYPE := Debug
else ifeq ($(SANITIZER),tsan)
  SAN_CMAKE_ARGS := -DENABLE_TSAN=ON
  BUILD_TYPE := Debug
else ifeq ($(SANITIZER),ubsan)
  SAN_CMAKE_ARGS := -DENABLE_UBSAN=ON
  BUILD_TYPE := Debug
else ifeq ($(SANITIZER),asan+ubsan)
  SAN_CMAKE_ARGS := -DENABLE_ASAN=ON -DENABLE_UBSAN=ON
  BUILD_TYPE := Debug
else ifeq ($(SANITIZER),tsan+ubsan)
  SAN_CMAKE_ARGS := -DENABLE_TSAN=ON -DENABLE_UBSAN=ON
  BUILD_TYPE := Debug
else
  $(error Unknown SANITIZER='$(SANITIZER)'. Use one of: asan, tsan, ubsan, asan+ubsan, tsan+ubsan)
endif

.PHONY: help image-build install build test generate-map launch-car launch-drone sh down clean gpu-image gpu-sh bench-build scenarios bench-cpp test-ml train

help:
	@echo "PathFinder make targets:"
	@echo "  image-build    Build the dev container image"
	@echo "  install        Run conan install inside the container"
	@echo "  build          colcon build inside the container"
	@echo "  test           colcon test inside the container"
	@echo "  generate-map   Generate maps/demo_world.bt"
	@echo "  launch-car     Full demo: car + sim + all four planners"
	@echo "  launch-drone   Full demo: drone + sim + all four planners"
	@echo "  sh             Open an interactive shell in the dev container"
	@echo "  down           Stop and remove containers"
	@echo "  clean          Stop containers and remove the local image"
	@echo "  gpu-image      Build the CUDA + PyTorch workbench image"
	@echo "  bench-build    Build the ROS-free benchmark harness"
	@echo "  scenarios      Generate the benchmark scenarios (bench/scenarios)"
	@echo "  bench-cpp      Benchmark the C++ planners -> bench/results/cpp.jsonl"
	@echo "  test-ml        Run the Python tests in the GPU container"
	@echo "  train          Train the RL planner with PPO (resumable) -> learning/runs/ppo"
	@echo ""
	@echo "  SANITIZER=<mode> on 'make build' enables a sanitizer (Debug build)."
	@echo "    Modes: asan, tsan, ubsan, asan+ubsan, tsan+ubsan"
	@echo "    Example: make build SANITIZER=asan"
	@echo ""
	@echo "  GPU=0 on the GPU-workbench targets runs them without a GPU."

image-build:
	$(COMPOSE) build dev

install:
	$(DEV) conan install . --output-folder=build --build=missing -s build_type=Release

build:
	$(DEV) bash -c 'source /opt/ros/jazzy/setup.bash && colcon build --cmake-args -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_TOOLCHAIN_FILE=$(TOOLCHAIN) -DCMAKE_PROJECT_INCLUDE=$(SANITIZERS_INCLUDE) $(SAN_CMAKE_ARGS)'

test:
	$(DEV) bash -c 'export $(SAN_RUNTIME) && source /opt/ros/jazzy/setup.bash && colcon test && colcon test-result --verbose'

generate-map:
	$(DEV) bash -c 'source /opt/ros/jazzy/setup.bash && source install/setup.bash && mkdir -p maps && ros2 run pathfinder_core generate_demo_map maps/demo_world.bt'

launch-car:
	$(DEV_TTY) bash -c 'export $(SAN_RUNTIME) && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch pathfinder_bringup demo_car.launch.py headless:=$(HEADLESS)'

launch-drone:
	$(DEV_TTY) bash -c 'export $(SAN_RUNTIME) && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 launch pathfinder_bringup demo_drone.launch.py headless:=$(HEADLESS)'

sh:
	$(DEV_TTY) bash

down:
	$(COMPOSE) down --remove-orphans

clean:
	$(COMPOSE) down --rmi local --remove-orphans

# ---------------------------------------------------------------------------
# GPU workbench (CUDA planners, RL planner, benchmarks). Runs in the `gpu`
# container. GPU=1 (default) hands the host GPU to the container, which needs
# the NVIDIA driver + nvidia-container-toolkit; GPU=0 runs everything on CPU.
# ---------------------------------------------------------------------------
GPU ?= 1
GPU_COMPOSE := $(COMPOSE) -f docker-compose.yml $(if $(filter 1,$(GPU)),-f docker-compose.gpu.yml,)
GPU_RUN := $(GPU_COMPOSE) run --rm -T gpu
GPU_RUN_TTY := $(GPU_COMPOSE) run --rm gpu

gpu-image:
	$(COMPOSE) build gpu

gpu-sh:
	$(GPU_RUN_TTY) bash

bench-build:
	$(GPU_RUN) bash -c 'cmake -S bench -B bench/build -G Ninja && cmake --build bench/build'

scenarios: bench-build
	$(GPU_RUN) bash -c './bench/build/pf_export_bt maps/demo_world.bt bench/build/demo_world.pfmap \
		&& python3 -m pathfinder_ml.scenarios --out bench/scenarios --demo bench/build/demo_world.pfmap'

bench-cpp: bench-build
	$(GPU_RUN) bash -c 'mkdir -p bench/results && ./bench/build/pf_bench --out bench/results/cpp.jsonl bench/scenarios/*.pfmap'

test-ml:
	$(GPU_RUN) python3 -m pytest -q learning/tests


# PPO training of the RL planner; resumable (re-run after an interruption).
TRAIN_ARGS ?=
train:
	$(GPU_RUN) python3 -m pathfinder_ml.train --out learning/runs/ppo --resume $(TRAIN_ARGS)
