#include <pathfinder_algo_rrt/rrt_core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace pathfinder_algo_rrt
{

namespace
{

constexpr std::size_t kMaxTreeEdgePoints = 1000;

struct Point3
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

inline double distance(const Point3 & a, const Point3 & b)
{
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

struct TreeNode
{
  Point3 position;
  int parent_idx{-1};
  double cost{0.0};
};

class RRTStar
{
public:
  RRTStar(const pathfinder_core::InflatedVoxelGrid & grid, const RRTParams & params, unsigned int seed)
  : grid_(grid), params_(params), rng_(seed)
  {
    const auto mn = grid_.min_bound();
    const auto mx = grid_.max_bound();
    min_bound_ = {mn[0], mn[1], mn[2]};
    max_bound_ = {mx[0], mx[1], mx[2]};
  }

  void add_node(const Point3 & p, int parent_idx, double cost)
  {
    tree_nodes_.push_back({p, parent_idx, cost});
  }

  const std::vector<TreeNode> & nodes() const { return tree_nodes_; }
  std::size_t size() const { return tree_nodes_.size(); }

  Point3 sample(const Point3 & goal)
  {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (unit(rng_) < params_.goal_bias) {
      return goal;
    }
    std::uniform_real_distribution<double> dx(min_bound_.x, max_bound_.x);
    std::uniform_real_distribution<double> dy(min_bound_.y, max_bound_.y);
    std::uniform_real_distribution<double> dz(min_bound_.z, max_bound_.z);
    return {dx(rng_), dy(rng_), dz(rng_)};
  }

  int nearest(const Point3 & q) const
  {
    int best = -1;
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < tree_nodes_.size(); ++i) {
      const double d = distance(tree_nodes_[i].position, q);
      if (d < best_d) {
        best_d = d;
        best = static_cast<int>(i);
      }
    }
    return best;
  }

  Point3 steer(const Point3 & from, const Point3 & to) const
  {
    const double d = distance(from, to);
    if (d <= params_.step_size || d < 1e-9) {
      return to;
    }
    const double s = params_.step_size / d;
    return {from.x + (to.x - from.x) * s,
            from.y + (to.y - from.y) * s,
            from.z + (to.z - from.z) * s};
  }

  std::vector<int> neighbors_within(const Point3 & q, double radius) const
  {
    std::vector<int> out;
    out.reserve(32);
    const double r2 = radius * radius;
    for (std::size_t i = 0; i < tree_nodes_.size(); ++i) {
      const double dx = tree_nodes_[i].position.x - q.x;
      const double dy = tree_nodes_[i].position.y - q.y;
      const double dz = tree_nodes_[i].position.z - q.z;
      if (dx * dx + dy * dy + dz * dz <= r2) {
        out.push_back(static_cast<int>(i));
      }
    }
    return out;
  }

  bool point_in_bounds(const Point3 & p) const
  {
    return p.x >= min_bound_.x && p.x <= max_bound_.x &&
           p.y >= min_bound_.y && p.y <= max_bound_.y &&
           p.z >= min_bound_.z && p.z <= max_bound_.z;
  }

  bool point_collision(const Point3 & p) const
  {
    return grid_.is_occupied_at_point(p.x, p.y, p.z);
  }

  bool segment_collision_free(const Point3 & a, const Point3 & b) const
  {
    if (!point_in_bounds(a) || !point_in_bounds(b)) {
      return false;
    }
    return grid_.is_segment_free(a.x, a.y, a.z, b.x, b.y, b.z);
  }

  int choose_parent(const std::vector<int> & near, const Point3 & x_new, int fallback_parent) const
  {
    int best = fallback_parent;
    double best_cost = (fallback_parent >= 0)
      ? tree_nodes_[fallback_parent].cost + distance(tree_nodes_[fallback_parent].position, x_new)
      : std::numeric_limits<double>::infinity();
    for (int idx : near) {
      if (idx == fallback_parent) {
        continue;
      }
      const double c = tree_nodes_[idx].cost + distance(tree_nodes_[idx].position, x_new);
      if (c < best_cost && segment_collision_free(tree_nodes_[idx].position, x_new)) {
        best_cost = c;
        best = idx;
      }
    }
    return best;
  }

  void rewire(const std::vector<int> & near, int new_idx)
  {
    const auto & x_new = tree_nodes_[new_idx];
    for (int idx : near) {
      if (idx == x_new.parent_idx) {
        continue;
      }
      auto & cand = tree_nodes_[idx];
      const double new_cost = x_new.cost + distance(x_new.position, cand.position);
      if (new_cost + 1e-9 < cand.cost &&
          segment_collision_free(x_new.position, cand.position))
      {
        cand.parent_idx = new_idx;
        const double delta = new_cost - cand.cost;
        cand.cost = new_cost;
        propagate_cost_decrease(idx, delta);
      }
    }
  }

  std::vector<Point3> reconstruct_from(int idx) const
  {
    std::vector<Point3> out;
    while (idx >= 0) {
      out.push_back(tree_nodes_[idx].position);
      idx = tree_nodes_[idx].parent_idx;
    }
    std::reverse(out.begin(), out.end());
    return out;
  }

private:
  void propagate_cost_decrease(int root_idx, double delta)
  {
    for (std::size_t i = 0; i < tree_nodes_.size(); ++i) {
      if (static_cast<int>(i) == root_idx) {
        continue;
      }
      int p = tree_nodes_[i].parent_idx;
      while (p >= 0) {
        if (p == root_idx) {
          // delta is negative on a cost decrease; add it so descendants drop too
          tree_nodes_[i].cost += delta;
          break;
        }
        p = tree_nodes_[p].parent_idx;
      }
    }
  }

  const pathfinder_core::InflatedVoxelGrid & grid_;
  RRTParams params_;
  std::mt19937 rng_;

  Point3 min_bound_;
  Point3 max_bound_;

  std::vector<TreeNode> tree_nodes_;
};

inline WorldPoint to_world_point(const Point3 & p)
{
  return WorldPoint{p.x, p.y, p.z};
}

std::vector<WorldPoint> path_to_world_points(const std::vector<Point3> & pts)
{
  std::vector<WorldPoint> out;
  out.reserve(pts.size());
  for (const auto & p : pts) {
    out.push_back(to_world_point(p));
  }
  return out;
}

std::vector<pathfinder_core::VoxelIndex> path_to_voxels(
  const std::vector<Point3> & pts,
  const pathfinder_core::InflatedVoxelGrid & grid)
{
  std::vector<pathfinder_core::VoxelIndex> out;
  out.reserve(pts.size());
  for (const auto & p : pts) {
    out.push_back(grid.world_to_voxel(p.x, p.y, p.z));
  }
  return out;
}

std::vector<std::pair<WorldPoint, WorldPoint>> sample_tree_edges(
  const std::vector<TreeNode> & nodes)
{
  std::vector<std::size_t> edge_indices;
  edge_indices.reserve(nodes.size());
  for (std::size_t i = 1; i < nodes.size(); ++i) {
    if (nodes[i].parent_idx >= 0) {
      edge_indices.push_back(i);
    }
  }
  const std::size_t max_edges = kMaxTreeEdgePoints / 2;
  if (edge_indices.size() > max_edges) {
    std::mt19937 rng(42);
    std::shuffle(edge_indices.begin(), edge_indices.end(), rng);
    edge_indices.resize(max_edges);
  }

  std::vector<std::pair<WorldPoint, WorldPoint>> edges;
  edges.reserve(edge_indices.size());
  for (auto i : edge_indices) {
    const auto & child = nodes[i];
    const auto & parent = nodes[child.parent_idx];
    edges.emplace_back(to_world_point(parent.position), to_world_point(child.position));
  }
  return edges;
}

// Tree path to the node that reaches the goal, plus the goal itself. The goal
// is never a tree node, so it cannot become an ancestor of its own parent.
std::vector<Point3> path_to_goal(const RRTStar & rrt, int goal_parent_idx, const Point3 & goal)
{
  auto pts = rrt.reconstruct_from(goal_parent_idx);
  if (pts.empty() || distance(pts.back(), goal) > 1e-9) {
    pts.push_back(goal);
  }
  return pts;
}

RRTFeedback build_feedback(
  const RRTStar & rrt,
  int goal_parent_idx,
  const Point3 & goal,
  double best_cost,
  const pathfinder_core::InflatedVoxelGrid & grid)
{
  RRTFeedback fb;
  fb.nodes_expanded = static_cast<int>(rrt.size());
  fb.best_cost_so_far = std::isfinite(best_cost)
    ? best_cost
    : std::numeric_limits<double>::infinity();

  std::vector<Point3> best_path;
  if (goal_parent_idx >= 0) {
    best_path = path_to_goal(rrt, goal_parent_idx, goal);
    fb.best_partial_path = path_to_voxels(best_path, grid);
    fb.best_partial_path_world = path_to_world_points(best_path);
  }
  fb.tree_edges = sample_tree_edges(rrt.nodes());
  return fb;
}

}  // namespace

RRTResult RRTCore::plan(
  const pathfinder_core::InflatedVoxelGrid & grid,
  const pathfinder_core::VoxelIndex & start,
  const pathfinder_core::VoxelIndex & goal,
  const RRTParams & params,
  FeedbackCallback feedback)
{
  RRTResult result;
  const auto t_start = std::chrono::steady_clock::now();

  const auto start_world = grid.voxel_to_world(start);
  const auto goal_world = grid.voxel_to_world(goal);
  const Point3 start_pt{start_world[0], start_world[1], start_world[2]};
  const Point3 goal_pt{goal_world[0], goal_world[1], goal_world[2]};

  const unsigned int seed = (params.random_seed > 0)
    ? params.random_seed
    : static_cast<unsigned int>(
        std::chrono::steady_clock::now().time_since_epoch().count());

  RRTStar rrt(grid, params, seed);

  if (rrt.point_collision(start_pt)) {
    result.success = false;
    result.message = "start in collision";
    return result;
  }
  if (rrt.point_collision(goal_pt)) {
    result.success = false;
    result.message = "goal in collision";
    return result;
  }

  rrt.add_node(start_pt, -1, 0.0);

  // The goal is tracked by the tree node that connects to it rather than added
  // as a node: as a node it could gain children, and re-parenting it onto one
  // of its own descendants later created a parent cycle that hung the planner.
  int goal_parent_idx = -1;
  double best_cost = std::numeric_limits<double>::infinity();
  int iter = 0;
  int iter_at_first_goal = -1;
  auto last_feedback = std::chrono::steady_clock::now();
  double slept_sec = 0.0;  // viz-delay sleeps must not count against the compute budget

  for (; iter < params.max_iterations; ++iter) {
    const auto now_t = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now_t - t_start).count();
    if (elapsed - slept_sec >= params.max_plan_time_sec) {
      break;
    }
    if (goal_parent_idx >= 0 &&
        (iter - iter_at_first_goal) >= params.min_iterations_after_goal)
    {
      break;
    }

    const Point3 x_rand = rrt.sample(goal_pt);
    const int nearest_idx = rrt.nearest(x_rand);
    if (nearest_idx < 0) {
      continue;
    }
    const Point3 x_new = rrt.steer(rrt.nodes()[nearest_idx].position, x_rand);
    if (!rrt.point_in_bounds(x_new)) {
      continue;
    }
    if (rrt.point_collision(x_new)) {
      continue;
    }
    if (!rrt.segment_collision_free(rrt.nodes()[nearest_idx].position, x_new)) {
      continue;
    }

    auto near = rrt.neighbors_within(x_new, params.rewire_radius);
    const int parent_idx = rrt.choose_parent(near, x_new, nearest_idx);
    if (parent_idx < 0) {
      continue;
    }
    const double cost_to_new =
      rrt.nodes()[parent_idx].cost + distance(rrt.nodes()[parent_idx].position, x_new);
    rrt.add_node(x_new, parent_idx, cost_to_new);
    const int new_idx = static_cast<int>(rrt.size()) - 1;
    rrt.rewire(near, new_idx);

    if (distance(x_new, goal_pt) <= params.goal_tolerance &&
        rrt.segment_collision_free(x_new, goal_pt))
    {
      const double candidate_cost = cost_to_new + distance(x_new, goal_pt);
      if (candidate_cost < best_cost) {
        if (goal_parent_idx < 0) {
          iter_at_first_goal = iter;
        }
        goal_parent_idx = new_idx;
        best_cost = candidate_cost;
      }
    }

    const double since_feedback =
      std::chrono::duration<double>(now_t - last_feedback).count();
    if (feedback && ((iter % 100) == 0 || since_feedback >= 0.2)) {
      feedback(build_feedback(rrt, goal_parent_idx, goal_pt, best_cost, grid));
      if (params.viz_delay_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(params.viz_delay_ms));
        slept_sec += params.viz_delay_ms / 1000.0;
      }
      last_feedback = now_t;
    }
  }

  const auto t_end = std::chrono::steady_clock::now();
  const double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  if (feedback) {
    feedback(build_feedback(rrt, goal_parent_idx, goal_pt, best_cost, grid));
  }

  result.plan_duration_ms = total_ms;
  result.nodes_expanded = static_cast<int>(rrt.size());

  if (goal_parent_idx >= 0) {
    const auto pts = path_to_goal(rrt, goal_parent_idx, goal_pt);
    result.path_world = path_to_world_points(pts);
    result.path = path_to_voxels(pts, grid);
    result.success = true;
    result.message = "ok";
  } else {
    result.success = false;
    result.message = "no path found in budget";
  }

  return result;
}

}  // namespace pathfinder_algo_rrt
