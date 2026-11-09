FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        wget \
        gnupg \
        lsb-release \
        ca-certificates \
    && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gz-harmonic \
        ros-jazzy-ros-gz \
        python3-numpy \
        python3-attr \
        python3-pil \
        python3-psutil \
        ros-jazzy-octomap-msgs \
        ros-jazzy-octomap-ros \
        cmake \
        ninja-build \
        python3-pip \
        python3-colcon-common-extensions \
        python3-rosdep \
    && pip install "conan>=2,<3" --break-system-packages \
    && rm -rf /var/lib/apt/lists/*

# rerun-sdk asks for numpy>=2, but the ROS message extensions are built against
# the distro numpy 1.26, so install it --no-deps. Its other deps come from apt
# above (attrs, pillow, psutil) or are pinned here.
RUN pip install --break-system-packages "pyarrow==21.0.0" "typing-extensions>=4.5" \
    && pip install --break-system-packages --no-deps "rerun-sdk==0.40.0"

RUN if ! id -u ros >/dev/null 2>&1; then \
        if getent passwd 1000 >/dev/null; then \
            existing=$(getent passwd 1000 | cut -d: -f1); \
            userdel -r "$existing" 2>/dev/null || true; \
        fi; \
        useradd -m -u 1000 -s /bin/bash ros; \
    fi

RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc

RUN conan profile detect --force

WORKDIR /workspace
