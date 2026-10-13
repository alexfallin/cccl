// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_EXECUTE_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_EXECUTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_merge.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cuda/__memory_resource/allocation_alignment.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/experimental/mgmn/__algorithm/common.h>
#include <cuda/experimental/mgmn/__algorithm/sort/select.h>

#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_HOST_API inline void __check(::cudaError_t __error)
{
  if (__error != ::cudaSuccess)
  {
    _CCCL_THROW(::cuda::cuda_error, __error, "executing distributed sort");
  }
}

template <class _Tp>
class __pinned_buffer
{
  _Tp* __ptr_ = nullptr;

public:
  _CCCL_HOST_API explicit __pinned_buffer(::cuda::std::size_t __count)
  {
    _CCCL_TRY_RUNTIME_API(::cudaMallocHost, "allocating sort metadata", &__ptr_, __count * sizeof(_Tp));
  }
  _CCCL_HOST_API ~__pinned_buffer()
  {
    _CCCL_ASSERT_RUNTIME_API(::cudaFreeHost, "freeing sort metadata", __ptr_);
  }
  __pinned_buffer(const __pinned_buffer&)            = delete;
  __pinned_buffer& operator=(const __pinned_buffer&) = delete;
  [[nodiscard]] _CCCL_HOST_API _Tp* __data() const noexcept
  {
    return __ptr_;
  }
  [[nodiscard]] _CCCL_HOST_API _Tp& operator[](::cuda::std::size_t __idx) const noexcept
  {
    return __ptr_[__idx];
  }
};

class __merge_streams
{
  ::std::vector<::cudaStream_t> __streams_;
  _CCCL_HOST_API void __release() noexcept
  {
    for (const auto __stream : __streams_)
    {
      // Join outstanding merges on exception paths before freeing their inputs.
      _CCCL_ASSERT_RUNTIME_API(::cudaStreamSynchronize, "waiting for sort merge", __stream);
      _CCCL_ASSERT_RUNTIME_API(::cudaStreamDestroy, "destroying sort merge stream", __stream);
    }
    __streams_.clear();
  }

public:
  _CCCL_HOST_API explicit __merge_streams(::cuda::std::size_t __count)
  {
    __streams_.reserve(__count);
    _CCCL_TRY
    {
      for (::cuda::std::size_t __idx = 0; __idx < __count; ++__idx)
      {
        ::cudaStream_t __stream;
        _CCCL_TRY_RUNTIME_API(
          ::cudaStreamCreateWithFlags, "creating sort merge stream", &__stream, cudaStreamNonBlocking);
        __streams_.push_back(__stream);
      }
    }
    _CCCL_CATCH_ALL
    {
      __release();
      _CCCL_RETHROW;
    }
  }
  _CCCL_HOST_API ~__merge_streams()
  {
    __release();
  }
  __merge_streams(const __merge_streams&)            = delete;
  __merge_streams& operator=(const __merge_streams&) = delete;
  [[nodiscard]] _CCCL_HOST_API ::cudaStream_t operator[](::cuda::std::size_t __idx) const noexcept
  {
    return __streams_[__idx];
  }
  _CCCL_HOST_API void __wait(::cuda::std::size_t __count) const
  {
    for (::cuda::std::size_t __idx = 0; __idx < __count; ++__idx)
    {
      _CCCL_TRY_RUNTIME_API(::cudaStreamSynchronize, "waiting for sort merge", __streams_[__idx]);
    }
  }
};

[[nodiscard]] _CCCL_HOST_API inline __index_type
__receive_offset(__index_type __begin, __index_type __count, __index_type& __cursor) noexcept
{
  if (__count == 0)
  {
    return __cursor;
  }
  const auto __offset    = __cursor + __begin % 16;
  __cursor               = __offset + __count;
  const auto __remainder = __cursor % 16;
  if (__remainder != 0)
  {
    __cursor += 16 - __remainder;
  }
  return __offset;
}

template <class _Comm, class _Tp>
using __all_gather_result_t = decltype(::cuda::std::declval<_Comm&>().all_gather(
  ::cuda::std::declval<typename ::cuda::std::remove_cvref_t<_Comm>::group_guard_type&>(),
  ::cuda::std::declval<const _Tp*>(),
  ::cuda::std::declval<_Tp*>(),
  ::cuda::std::declval<::cuda::std::size_t>(),
  ::cuda::std::declval<::cuda::stream_ref>()));

template <class _Comm, class _Tp>
_CCCL_HOST_API void __all_gather(
  _Comm& __comm, const _Tp* __source, _Tp* __result, ::cuda::std::size_t __count, ::cuda::stream_ref __stream)
{
  auto __guard = __comm.group_guard();
  if constexpr (::cuda::std::_IsValidExpansion<__all_gather_result_t, _Comm, _Tp>::value)
  {
    __comm.all_gather(__guard, __source, __result, __count, __stream);
  }
  else
  {
    for (int __peer = 0; __peer < __comm.size(); ++__peer)
    {
      if (__peer == __comm.rank())
      {
        ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
          __result + __count * __peer, __source, __count * sizeof(_Tp), ::cudaMemcpyDeviceToDevice, __stream.get()));
      }
      else
      {
        __comm.send(__guard, __source, __count, __peer, __stream);
        __comm.recv(__guard, __result + __count * __peer, __count, __peer, __stream);
      }
    }
  }
}

template <class _Key, class _Compare, class _Env>
_CCCL_HOST_API void
__local_sort(_Key* __keys, _Key* __sorted_output, ::cuda::std::int64_t __count, _Compare __cmp, const _Env& __env)
{
  if (__count == 0)
  {
    return;
  }
  if constexpr (::cuda::std::is_arithmetic_v<_Key>
                && (::cuda::std::is_same_v<_Compare, ::cuda::std::less<>>
                    || ::cuda::std::is_same_v<_Compare, ::cuda::std::less<_Key>>) )
  {
    ::cuda::experimental::mgmn::__detail::__sort::__check(
      CUB_NS_QUALIFIER::DeviceRadixSort::SortKeys(__keys, __sorted_output, __count, 0, sizeof(_Key) * 8, __env));
  }
  else if constexpr (::cuda::std::is_arithmetic_v<_Key>
                     && (::cuda::std::is_same_v<_Compare, ::cuda::std::greater<>>
                         || ::cuda::std::is_same_v<_Compare, ::cuda::std::greater<_Key>>) )
  {
    ::cuda::experimental::mgmn::__detail::__sort::__check(CUB_NS_QUALIFIER::DeviceRadixSort::SortKeysDescending(
      __keys, __sorted_output, __count, 0, sizeof(_Key) * 8, __env));
  }
  else
  {
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
      __sorted_output, __keys, __count * sizeof(_Key), ::cudaMemcpyDeviceToDevice, ::cuda::get_stream(__env).get()));
    ::cuda::experimental::mgmn::__detail::__sort::__check(
      CUB_NS_QUALIFIER::DeviceMergeSort::SortKeys(__sorted_output, __count, __cmp, __env));
  }
}

template <class _Env>
class __scratch_allocator
{
  const _Env& __env_;
  ::cuda::stream_ref __stream_;
  ::cuda::experimental::mgmn::__detail::__resource_type_for<_Env> __resource_;

public:
  _CCCL_HOST_API __scratch_allocator(const _Env& __env, ::cuda::stream_ref __stream)
      : __env_(__env)
      , __stream_(__stream)
      , __resource_(::cuda::experimental::mgmn::__detail::__resource_from_env(__env, __stream.__logical_device()))
  {}
  template <class _Tp>
  [[nodiscard]] _CCCL_HOST_API auto operator()(::cuda::std::size_t __count) const
  {
    // Sixteen-key offsets preserve the transfer alignment requested here.
    const auto __requested = ::cuda::__call_or(::cuda::allocation_alignment, alignof(_Tp), __env_);
    ::cuda::__validate_allocation_alignment(__requested, alignof(_Tp));
    using _Alignment        = ::cuda::std::execution::prop<::cuda::allocation_alignment_t, ::cuda::std::size_t>;
    const auto __buffer_env = ::cuda::std::execution::env<_Alignment, const _Env&>{
      _Alignment{::cuda::allocation_alignment, ::cuda::std::max(::cuda::std::size_t{128}, __requested)}, __env_};
    return ::cuda::experimental::mgmn::__detail::__make_safe_uninitialized_buffer<_Tp>(
      __stream_, __resource_, __count, __buffer_env);
  }
};

template <class _Comm, class _Env, class _Key, class _Size, class _Compare>
_CCCL_HOST_API void __sort_one(_Comm& __comm, const _Env& __env, _Key* __keys, _Size __input_count, _Compare __cmp)
{
  static_assert(::cuda::std::is_integral_v<_Size>, "sort count must be an integer");
  const ::cuda::stream_ref __stream = ::cuda::get_stream(__env);
  const auto __context              = ::cuda::__ensure_current_context{__stream};
  const int __rank = __comm.rank(), __ranks = __comm.size();
  if (__ranks <= 0 || __rank < 0 || __rank >= __ranks)
  {
    _CCCL_THROW(::std::invalid_argument, "Invalid communicator rank/size");
  }
  const __scratch_allocator<_Env> __allocate{__env, __stream};

  // Agree on the caller's fixed output counts before selection. Invalid local
  // counts are exchanged too, so a negative count cannot strand other ranks.
  const ::cuda::std::uint64_t __limit =
    static_cast<::cuda::std::uint64_t>(::cuda::std::numeric_limits<::cuda::std::int64_t>::max()) / sizeof(_Key);
  const ::cuda::std::uint64_t __padding = ::cuda::std::uint64_t{30} * __ranks;
  const bool __invalid =
    (::cuda::std::is_unsigned_v<_Size> ? false : (__input_count < 0))
    || static_cast<::cuda::std::uint64_t>(__input_count) > __limit - __padding
    || (__input_count != 0 && __keys == nullptr);
  __pinned_buffer<long long> __counts{static_cast<::cuda::std::size_t>(__ranks) + 1};
  __counts[__ranks]    = __invalid ? -1 : static_cast<long long>(__input_count);
  auto __source_count  = __allocate.template operator()<long long>(1);
  auto __device_counts = __allocate.template operator()<long long>(__ranks);
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __source_count.data(), __counts.__data() + __ranks, sizeof(long long), ::cudaMemcpyHostToDevice, __stream.get()));
  ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
    __comm, __source_count.data(), __device_counts.data(), 1, __stream);
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __counts.__data(), __device_counts.data(), __ranks * sizeof(long long), ::cudaMemcpyDeviceToHost, __stream.get()));
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  ::std::vector<::cuda::std::int64_t> __targets(__ranks + 1, 0);
  bool __uniform = true;
  for (int __r = 0; __r < __ranks; ++__r)
  {
    if (__counts[__r] < 0 || __counts[__r] > ::cuda::std::numeric_limits<::cuda::std::int64_t>::max() - __targets[__r])
    {
      _CCCL_THROW(::std::invalid_argument, "Invalid distributed sort input count or total size overflow");
    }
    __targets[__r + 1] = __targets[__r] + __counts[__r];
    __uniform &= __counts[__r] == __counts[0];
  }
  if (__targets.back() == 0)
  {
    return;
  }
  const ::cuda::std::int64_t __size = __counts[__rank];
  auto __sorted = __allocate.template operator()<_Key>(::cuda::std::max<::cuda::std::int64_t>(__size, 1));
  ::cuda::experimental::mgmn::__detail::__sort::__local_sort(__keys, __sorted.data(), __size, __cmp, __env);
  if (__ranks == 1)
  {
    ::cuda::experimental::mgmn::__detail::__sort::__check(
      ::cudaMemcpyAsync(__keys, __sorted.data(), __size * sizeof(_Key), ::cudaMemcpyDeviceToDevice, __stream.get()));
    return;
  }

  const ::cuda::std::size_t __width = static_cast<::cuda::std::size_t>(__ranks) + 1;
  auto __source                     = __allocate.template operator()<__selection_message<_Key>>(__width);
  auto __gathered                   = __allocate.template operator()<__selection_message<_Key>>(__width * __ranks);
  auto __states                     = __allocate.template operator()<__selection_state<_Key>>(__ranks + 1);
  auto __device_targets             = __allocate.template operator()<::cuda::std::int64_t>(__ranks + 1);
  auto __unfinished                 = __allocate.template operator()<int>(1);
  __pinned_buffer<int> __host_unfinished{1};
  __pinned_buffer<::cuda::std::int64_t> __cuts{static_cast<::cuda::std::size_t>(__ranks + 1) * __ranks};
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __device_targets.data(),
    __targets.data(),
    (__ranks + 1) * sizeof(::cuda::std::int64_t),
    ::cudaMemcpyHostToDevice,
    __stream.get()));
  auto __local_cuts    = __allocate.template operator()<::cuda::std::int64_t>(__width);
  auto __gathered_cuts = __allocate.template operator()<::cuda::std::int64_t>(__width * __ranks);
  auto __step          = [&](auto __phase) {
    ::cuda::experimental::mgmn::__detail::__sort::__select<decltype(__phase)::value><<<1, 256, 0, __stream.get()>>>(
      __sorted.data(),
      __size,
      __rank,
      __ranks,
      __source.data(),
      __gathered.data(),
      __states.data(),
      __unfinished.data(),
      __cmp,
      __device_targets.data(),
      __uniform,
      __local_cuts.data());
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaGetLastError());
  };
  using __phase = __selection_phase;
  __step(::cuda::std::integral_constant<__phase, __phase::__initialize>{});
  ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
    __comm, __source.data(), __gathered.data(), __width, __stream);
  __step(::cuda::std::integral_constant<__phase, __phase::__first_count>{});
  do
  {
    ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
      __comm, __source.data(), __gathered.data(), __width, __stream);
    __step(::cuda::std::integral_constant<__phase, __phase::__advance>{});
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
      __host_unfinished.__data(), __unfinished.data(), sizeof(int), ::cudaMemcpyDeviceToHost, __stream.get()));
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  } while (__host_unfinished[0]);
  __step(::cuda::std::integral_constant<__phase, __phase::__export_cuts>{});
  ::cuda::experimental::mgmn::__detail::__sort::__all_gather(
    __comm, __local_cuts.data(), __gathered_cuts.data(), __ranks + 1, __stream);
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
    __cuts.__data(),
    __gathered_cuts.data(),
    ::cuda::std::size_t(__ranks + 1) * __ranks * sizeof(long long),
    ::cudaMemcpyDeviceToHost,
    __stream.get()));
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  auto __cut = [&](int __boundary, int __peer) -> ::cuda::std::int64_t {
    return __cuts[::cuda::std::size_t(__peer) * (__ranks + 1) + __boundary];
  };

  auto __received = __allocate.template operator()<_Key>(__size + __padding);
  auto __scratch  = __allocate.template operator()<_Key>(::cuda::std::max<::cuda::std::int64_t>(__size, 1));
  struct __run
  {
    ::cuda::std::int64_t __offset_, __count_;
  };
  ::std::vector<__run> __runs;
  ::cuda::std::int64_t __cursor = 0;
  {
    auto __guard = __comm.group_guard();
    for (int __peer = 0; __peer < __ranks; ++__peer)
    {
      const ::cuda::std::int64_t __begin = __cut(__rank, __peer), __count = __cut(__rank + 1, __peer) - __begin;
      const ::cuda::std::int64_t __offset =
        ::cuda::experimental::mgmn::__detail::__sort::__receive_offset(__begin, __count, __cursor);
      if (__count)
      {
        __runs.push_back({__offset, __count});
      }
      if (__peer == __rank)
      {
        if (__count)
        {
          ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
            __received.data() + __offset,
            __sorted.data() + __begin,
            __count * sizeof(_Key),
            ::cudaMemcpyDeviceToDevice,
            __stream.get()));
        }
      }
      else
      {
        const ::cuda::std::int64_t __send_begin  = __cut(__peer, __rank),
                                   __send_count  = __cut(__peer + 1, __rank) - __send_begin;
        const ::cuda::std::int64_t __send_prefix = __send_begin % 16;
        const ::cuda::std::int64_t __recv_prefix = __begin % 16;
        if (__send_count)
        {
          __comm.send(
            __guard, __sorted.data() + __send_begin - __send_prefix, __send_count + __send_prefix, __peer, __stream);
        }
        if (__count)
        {
          __comm.recv(__guard, __received.data() + __offset - __recv_prefix, __count + __recv_prefix, __peer, __stream);
        }
      }
    }
  }
  // Makes receive data and async scratch allocations visible to merge streams.
  ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaStreamSynchronize(__stream.get()));
  _Key* __merge_input  = __received.data();
  _Key* __merge_output = __scratch.data();
  __merge_streams __streams{__runs.size() / 2};
  while (__runs.size() > 1)
  {
    ::std::vector<__run> __next;
    ::cuda::std::int64_t __sorted_output_offset = 0;
    const ::cuda::std::size_t __pairs           = __runs.size() / 2;
    for (::cuda::std::size_t __pair = 0; __pair < __pairs; ++__pair)
    {
      const auto __a = __runs[2 * __pair], __b = __runs[2 * __pair + 1];
      // Override only the stream; preserve resource, tuning and all other
      // CUB environment queries. Each concurrent merge owns its scratch.
      auto __merge_env =
        ::cuda::std::execution::env<::cuda::stream_ref, const _Env&>{::cuda::stream_ref{__streams[__pair]}, __env};
      ::cuda::experimental::mgmn::__detail::__sort::__check(CUB_NS_QUALIFIER::DeviceMerge::MergeKeys(
        __merge_input + __a.__offset_,
        __a.__count_,
        __merge_input + __b.__offset_,
        __b.__count_,
        __merge_output + __sorted_output_offset,
        __cmp,
        __merge_env));
      __next.push_back({__sorted_output_offset, __a.__count_ + __b.__count_});
      __sorted_output_offset += __a.__count_ + __b.__count_;
    }
    if (__runs.size() % 2)
    {
      const auto __tail = __runs.back();
      ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
        __merge_output + __sorted_output_offset,
        __merge_input + __tail.__offset_,
        __tail.__count_ * sizeof(_Key),
        ::cudaMemcpyDeviceToDevice,
        __streams[0]));
      __next.push_back({__sorted_output_offset, __tail.__count_});
    }
    __streams.__wait(__pairs);
    __runs.swap(__next);
    ::cuda::std::swap(__merge_input, __merge_output);
  }
  if (__size)
  {
    ::cuda::experimental::mgmn::__detail::__sort::__check(::cudaMemcpyAsync(
      __keys, __merge_input + __runs[0].__offset_, __size * sizeof(_Key), ::cudaMemcpyDeviceToDevice, __stream.get()));
  }
  // Buffers free on the caller stream after the final copy. The caller can
  // enqueue consumers there immediately, or synchronize that stream to wait.
}
template <class _CommRange, class _EnvRange, class _InputRange, class _SizeRange, class _Compare = ::cuda::std::less<>>
_CCCL_HOST_API void __execute_range(
  _CommRange&& __comms, _EnvRange&& __envs, _InputRange&& __inputs, _SizeRange&& __sizes, _Compare __cmp = {})
{
  const auto __count = ::cuda::std::ranges::distance(__comms);
  if (__count == 0)
  {
    return;
  }
  if (::cuda::std::ranges::distance(__envs) != __count || ::cuda::std::ranges::distance(__inputs) != __count
      || ::cuda::std::ranges::distance(__sizes) != __count)
  {
    _CCCL_THROW(::std::invalid_argument, "Distributed sort ranges must have the same length");
  }
  auto __c = ::cuda::std::ranges::begin(__comms);
  auto __e = ::cuda::std::ranges::begin(__envs);
  auto __i = ::cuda::std::ranges::begin(__inputs);
  auto __n = ::cuda::std::ranges::begin(__sizes);
  if (__count == 1)
  {
    ::cuda::experimental::mgmn::__detail::__sort::__sort_one(*__c, *__e, ::cuda::std::to_address(*__i), *__n, __cmp);
    return;
  }
  // Blocking metadata decisions require concurrent participation by local
  // ranks. Each worker has independent state and selects its stream's device.
  ::std::vector<::std::exception_ptr> __errors(__count);
  ::std::vector<::std::thread> __workers;
  __workers.reserve(__count);
  ::std::mutex __mutex;
  ::std::condition_variable __ready;
  int __start = 0;
  _CCCL_TRY
  {
    for (::cuda::std::remove_cv_t<decltype(__count)> __r = 0; __r < __count; ++__r, ++__c, ++__e, ++__i, ++__n)
    {
      __workers.emplace_back([&, __c, __e, __i, __n, __r, __cmp] {
        {
          ::std::unique_lock<::std::mutex> __lock{__mutex};
          __ready.wait(__lock, [&] {
            return __start != 0;
          });
          if (__start == 2)
          {
            return;
          }
        }
        _CCCL_TRY
        {
          ::cuda::experimental::mgmn::__detail::__sort::__sort_one(
            *__c, *__e, ::cuda::std::to_address(*__i), *__n, __cmp);
        }
        _CCCL_CATCH_ALL
        {
          __errors[__r] = ::std::current_exception();
        }
      });
    }
  }
  _CCCL_CATCH_ALL
  {
    // Do not leave already-created workers inside a collective when a
    // later host thread cannot be started.
    {
      const ::std::lock_guard<::std::mutex> __lock{__mutex};
      __start = 2;
    }
    __ready.notify_all();
    for (auto& __worker : __workers)
    {
      __worker.join();
    }
    _CCCL_RETHROW;
  }
  {
    const ::std::lock_guard<::std::mutex> __lock{__mutex};
    __start = 1;
  }
  __ready.notify_all();
  for (auto& __worker : __workers)
  {
    __worker.join();
  }
  for (auto __error : __errors)
  {
    if (__error)
    {
      ::std::rethrow_exception(__error);
    }
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_EXECUTE_H
