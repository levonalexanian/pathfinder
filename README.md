# PathFinder

ROS2 Jazzy C++ showcase of four pathfinding algorithms — A\*, Dijkstra, RRT\*, D\* Lite — planning through a 3D voxel map in Gazebo Harmonic, visualized in [Rerun](https://rerun.io). Two robots: a holonomic drone and a diff-drive car.

## Requirements

- **Docker** with Compose
- **make**
- A modern web browser

ROS2 Jazzy, Gazebo Harmonic, Conan 2, and the C++ toolchain all run in the dev container. The Rerun web viewer is served from the container too — no host install needed.

## First-time setup

```bash
git clone https://github.com/levonalexanian/PathFinder.git
cd PathFinder
make image-build      # ~5-10 min, one-time
make install
make build
make generate-map
```

## Run the demo

```bash
make launch-car     # diff-drive car
make launch-drone   # holonomic drone
```

Then open **http://localhost:9090** for the Rerun web viewer. The `rerun_bridge` node (`pathfinder_rerun` package) logs the map, robot URDF and pose, goal, planned path, live search state and planner timings into one recording, with a default layout (3D scene, plan-duration plot, event log). Scrub the timeline to replay a plan.

Prefer the native viewer? Point it at the bridge: `rerun rerun+http://localhost:9876/proxy`. Pass `launch_rerun:=false` to skip the bridge entirely. To record a session instead, run the bridge yourself with `-p mode:=save -p save_path:=run.rrd` and open the file in any Rerun viewer.

### Send a goal

In another terminal:

```bash
make sh
source install/setup.bash
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: 9.0, y: 9.0, z: 1.5}}}'
```

### Switch algorithms

```bash
ros2 topic pub --once /algorithm_selection pathfinder_msgs/AlgorithmSelection \
  '{algorithm_name: dijkstra}'    # astar, dijkstra, rrt, or dstar_lite
```

Dijkstra explores ~35× more nodes than A\* on the same path; D\* Lite returns near-instantly on its second call (incremental replan).

### Watching the search

While the planner runs, it publishes a `MarkerArray` on `/search_visualization` that lets you watch the algorithm think. Each feedback frame replaces the previous one, and the scheduler clears the search a couple of seconds after planning ends (just before the robot starts moving); scrub back on the timeline to see any earlier frame.

- **Yellow cubes** — the *open set*: voxels the algorithm is considering but hasn't expanded yet (the "frontier"). For A\* / Dijkstra / D\* Lite. For RRT\*, these are the latest tree-edge segments.
- **Grey cubes** — the *closed set* (or "locked" for D\* Lite): voxels the algorithm has already expanded and won't revisit. Watch how Dijkstra's grey region spreads symmetrically outward while A\* leans toward the goal — that's the heuristic at work.
- **Green line** — the current best path (`CURRENT_BEST_PATH`): refreshed each feedback frame, replaced when a better path is found.

The number of cubes is capped (~500 per type per frame) so the viewer stays responsive; the visualization is indicative, not exhaustive.

## Make targets

| Target | What |
|---|---|
| `image-build`, `install`, `build`, `test` | Container, deps, colcon |
| `generate-map` | Regenerate `maps/demo_world.bt` |
| `launch-{car,drone}` | Full demo (robot + sim + all four planners) |
| `sh` | Shell in the dev container (for `ros2 launch …` of individual planners, etc.) |
| `down`, `clean` | Stop containers / remove image |

### Sanitizer builds

`make build` accepts a `SANITIZER` variable that injects the appropriate `-fsanitize=...` flags into every package and switches to a `Debug` build for cleaner stack traces:

```bash
make build SANITIZER=asan          # AddressSanitizer
make build SANITIZER=tsan          # ThreadSanitizer
make build SANITIZER=ubsan         # UndefinedBehaviorSanitizer
make build SANITIZER=asan+ubsan    # ASan + UBSan together
make build SANITIZER=tsan+ubsan    # TSan + UBSan together
```

ASan and TSan are mutually exclusive — combining them is a configure-time fatal error. `MemorySanitizer` is **not** supported, since it would require rebuilding ROS 2, libstdc++, and every transitive library with MSan instrumentation.

Useful runtime env vars (set before `make launch-*` or `ros2 run`):

```bash
export ASAN_OPTIONS=symbolize=1:print_stacktrace=1
export TSAN_OPTIONS=second_deadlock_stack=1
```

LeakSanitizer is bundled into ASan and will likely report leaks coming from ROS 2 / Gazebo initialization that are not in your code. Silence them with `ASAN_OPTIONS=detect_leaks=0` once you've confirmed they aren't yours.

## Layout

```
src/
├── algorithms/   pathfinder_algo_{astar,dijkstra,dstar_lite,rrt}
├── robots/       pathfinder_robot_{car,drone}
└── infra/        pathfinder_{msgs,core,bringup}
```

Each algorithm follows a Node/Core split: a pure C++ library (`*_core`) holds the algorithm and is unit-testable without ROS; a thin ROS Node (`*_node`) wraps it. Parameters live in `config/params.yaml` per package.

CI builds and tests on push and PR.
