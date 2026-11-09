# GPU workbench: CUDA toolchain + PyTorch for the CUDA planners, the RL planner
# and the benchmark harness. Independent of the ROS dev image.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        liboctomap-dev \
        pkg-config \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1+cu130" \
    && pip install "numpy>=1.26,<3" "rerun-sdk==0.40.0" "pytest>=8,<9" "matplotlib>=3.8,<4"

ENV HOME=/tmp/home
WORKDIR /workspace
