#pragma once

// Reader/writer for the .pfmap scenario format (see learning/pathfinder_ml/grid.py).

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pf_bench
{

struct Query
{
  std::array<int, 3> start;
  std::array<int, 3> goal;
};

struct PfMap
{
  std::string name;
  std::array<int, 3> dims{0, 0, 0};  // nx, ny, nz
  std::array<double, 3> min_bound{0.0, 0.0, 0.0};
  double resolution{0.1};
  std::vector<std::uint8_t> occupied;  // flat (z*ny + y)*nx + x
  std::vector<Query> queries;
};

inline PfMap load_pfmap(const std::string & path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open " + path);
  }
  char magic[8];
  f.read(magic, 8);
  if (std::memcmp(magic, "PFMAP001", 8) != 0) {
    throw std::runtime_error(path + ": bad magic");
  }
  PfMap m;
  f.read(reinterpret_cast<char *>(m.dims.data()), 12);
  f.read(reinterpret_cast<char *>(m.min_bound.data()), 24);
  f.read(reinterpret_cast<char *>(&m.resolution), 8);
  const std::size_t n = static_cast<std::size_t>(m.dims[0]) * m.dims[1] * m.dims[2];
  m.occupied.resize(n);
  f.read(reinterpret_cast<char *>(m.occupied.data()), static_cast<std::streamsize>(n));
  std::int32_t nq = 0;
  f.read(reinterpret_cast<char *>(&nq), 4);
  for (int i = 0; i < nq; ++i) {
    std::int32_t q[6];
    f.read(reinterpret_cast<char *>(q), 24);
    m.queries.push_back({{q[0], q[1], q[2]}, {q[3], q[4], q[5]}});
  }
  if (!f) {
    throw std::runtime_error(path + ": truncated");
  }
  const auto slash = path.find_last_of('/');
  const auto base = path.substr(slash == std::string::npos ? 0 : slash + 1);
  m.name = base.substr(0, base.rfind('.'));
  return m;
}

inline void save_pfmap(const std::string & path, const PfMap & m)
{
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot write " + path);
  }
  f.write("PFMAP001", 8);
  f.write(reinterpret_cast<const char *>(m.dims.data()), 12);
  f.write(reinterpret_cast<const char *>(m.min_bound.data()), 24);
  f.write(reinterpret_cast<const char *>(&m.resolution), 8);
  f.write(reinterpret_cast<const char *>(m.occupied.data()),
    static_cast<std::streamsize>(m.occupied.size()));
  const std::int32_t nq = static_cast<std::int32_t>(m.queries.size());
  f.write(reinterpret_cast<const char *>(&nq), 4);
  for (const auto & q : m.queries) {
    const std::int32_t v[6] = {q.start[0], q.start[1], q.start[2], q.goal[0], q.goal[1], q.goal[2]};
    f.write(reinterpret_cast<const char *>(v), 24);
  }
}

}  // namespace pf_bench
