// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/buffer>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/span>

#include <cuda/experimental/mgmn/sort>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>
#include <testing.cuh>

#include "sort_common.cuh"
#include <c2h/catch2_test_helper.h>

namespace
{
// Exercise the public communicator concept without requiring optional collectives.
struct point_to_point_comm
{
  using native_handle_type = cudax::mgmn::nccl_communicator_ref::native_handle_type;
  using group_guard_type   = cudax::mgmn::nccl_communicator_ref::group_guard_type;
  cudax::mgmn::nccl_communicator_ref comm;
  native_handle_type native_handle() const noexcept
  {
    return comm.native_handle();
  }
  auto rank() const
  {
    return comm.rank();
  }
  auto size() const
  {
    return comm.size();
  }
  auto group_guard() const
  {
    return comm.group_guard();
  }
  template <class T>
  void send(group_guard_type& guard, T* data, cuda::std::size_t count, int peer, cuda::stream_ref stream) const
  {
    comm.send(guard, data, count, peer, stream);
  }
  template <class T>
  void recv(group_guard_type& guard, T* data, cuda::std::size_t count, int peer, cuda::stream_ref stream) const
  {
    comm.recv(guard, data, count, peer, stream);
  }
};

struct masked_less
{
  cuda::std::uint32_t mask;
  __host__ __device__ bool operator()(int left, int right) const
  {
    return (static_cast<cuda::std::uint32_t>(left) ^ mask) < (static_cast<cuda::std::uint32_t>(right) ^ mask);
  }
};
} // namespace

MULTI_GPU_TEST("sort, fixed subranges and pending stream work", )
{
  auto comms         = this->communicators();
  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};
  std::vector<std::vector<int>> host_inputs(comms.size());
  std::vector<int> expected;
  std::vector<cuda::std::int64_t> counts;
  constexpr int guard = 0x7fffffff;
  constexpr masked_less cmp{0x555aaaaU};
  for (cuda::std::size_t rank = 0; rank < comms.size(); ++rank)
  {
    const auto count = static_cast<cuda::std::int64_t>(rank % 3 == 1 ? 0 : 73 + 257 * rank);
    counts.push_back(count);
    auto& local = host_inputs[rank];
    local.push_back(guard);
    for (cuda::std::int64_t i = 0; i < count; ++i)
    {
      const auto key = static_cast<int>((i * 917 + rank * 61) % 257) - 128;
      local.push_back(key);
      expected.push_back(key);
    }
    local.push_back(guard);
  }
  std::sort(expected.begin(), expected.end(), cmp);
  auto buffers = sort_test_util::make_device_inputs(comms, streams, host_inputs);
  std::vector<int*> inputs;
  for (auto& buffer : buffers)
  {
    inputs.push_back(buffer.data() + 1);
  }
  SECTION("native all-gather")
  {
    cudax::mgmn::sort(cudax::distributed, comms, streams, inputs, counts, cmp);
  }
  SECTION("point-to-point all-gather")
  {
    std::vector<point_to_point_comm> wrappers;
    for (auto comm : comms)
    {
      wrappers.push_back({comm});
    }
    cudax::mgmn::sort(cudax::distributed, wrappers, streams, inputs, counts, cmp);
  }
  // Gathering immediately on the supplied streams also checks final-copy ordering.
  const auto output                 = sort_test_util::gather_outputs(comms, buffers);
  cuda::std::size_t offset          = 0;
  cuda::std::size_t expected_offset = 0;
  for (const auto count : counts)
  {
    REQUIRE(output[offset] == guard);
    REQUIRE(output[offset + count + 1] == guard);
    REQUIRE(
      std::equal(output.begin() + offset + 1, output.begin() + offset + count + 1, expected.begin() + expected_offset));
    offset += count + 2;
    expected_offset += count;
  }
}

MULTI_GPU_TEST("sort, rejects invalid fixed ranges collectively", )
{
  auto comms         = this->communicators();
  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};
  std::vector<int*> inputs(comms.size(), nullptr);
  std::vector<cuda::std::int64_t> counts(comms.size(), 0);
  REQUIRE_NOTHROW(cudax::mgmn::sort(cudax::distributed, comms, streams, inputs, counts));
  if (comms.empty())
  {
    return;
  }
  SECTION("negative count on one rank")
  {
    counts.back() = -1;
  }
  SECTION("nonempty null input on one rank")
  {
    counts.back() = 1;
  }
  SECTION("byte count overflow on one rank")
  {
    counts.back() = std::numeric_limits<cuda::std::int64_t>::max();
  }
  REQUIRE_THROWS_AS(cudax::mgmn::sort(cudax::distributed, comms, streams, inputs, counts), std::invalid_argument);
}

TEST_CASE("sort, empty communicator range is a no-op")
{
  std::vector<cudax::mgmn::nccl_communicator_ref> comms;
  std::vector<cuda::stream_ref> streams;
  std::vector<int*> inputs{nullptr};
  std::vector<int> counts;
  REQUIRE_NOTHROW(cudax::mgmn::sort(cudax::distributed, comms, streams, inputs, counts));
}

MULTI_GPU_TEST("sort, rejects mismatched nonempty outer ranges", )
{
  auto comms         = this->communicators();
  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};
  std::vector<int*> inputs(comms.size(), nullptr);
  std::vector<int> counts(comms.size(), 0);
  streams.pop_back();
  REQUIRE_THROWS_AS(cudax::mgmn::sort(cudax::distributed, comms, streams, inputs, counts), std::invalid_argument);
}
