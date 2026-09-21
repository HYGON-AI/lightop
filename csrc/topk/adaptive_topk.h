#pragma once
// Exact Top512/1024/2048 over fully materialized FP32 scores, for wave64 HIP.
// Sampling only proposes a cutoff; a complete scan certifies every fast path.
// All decisions remain on device during Graph replay.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>
#include <climits>

namespace at {
namespace native {
namespace adaptive_topk_detail {
constexpr int Threads = 1024;
constexpr int Samples = 2048;
constexpr int SampleItems = Samples / Threads;
using SampleSort = hipcub::BlockRadixSort<uint32_t, Threads, SampleItems>;
using Scan = hipcub::BlockScan<int, Threads>;
using Reduce = hipcub::BlockReduce<int, Threads>;
struct Counts { int positive, zero, nan; };
struct AddCounts {
  __device__ Counts operator()(const Counts& a, const Counts& b) const {
    return {a.positive+b.positive, a.zero+b.zero, a.nan+b.nan};
  }
};
using CountReduce = hipcub::BlockReduce<Counts, Threads>;
struct KeyRange { uint32_t low, high; };
struct MergeRange {
  __device__ KeyRange operator()(const KeyRange& a, const KeyRange& b) const {
    return {::min(a.low,b.low), ::max(a.high,b.high)};
  }
};
using RangeReduce = hipcub::BlockReduce<KeyRange, Threads>;

union Temporary {
  Scan::TempStorage scan;
  Reduce::TempStorage reduce;
  CountReduce::TempStorage counts;
  RangeReduce::TempStorage range;
};

template <int K>
struct Shared {
  // These lifetimes do not overlap: propose, then cache/refine.
  union {
    struct { uint32_t keys[2*K]; int ids[2*K]; };
    SampleSort::TempStorage sample;
  };
  int selected[((K+Threads-1)/Threads)*Threads];
  int hist[256];
  Temporary temp;
  uint32_t pivot, prefix, mask;
  uint32_t low, high;
  int count, equal, rank, active, route, emitted, eq_used;
};

// Matches LightOp's ordered FP32 bits, including signed zero / NaN payloads.
__device__ __forceinline__ uint32_t key(float value) {
  uint32_t bits = __float_as_uint(value);
  return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}

__device__ __forceinline__ int reserve_wave(bool keep, int* counter) {
  const int lane = threadIdx.x % 64;
  const uint64_t votes = __ballot(keep);
  if (!votes) return 0;
  int base = 0;
  if (lane == 0 && votes) base = atomicAdd(counter, __popcll(votes));
  base = __shfl(base, 0, 64);
  const uint64_t before = lane ? ((uint64_t{1} << lane) - 1) : 0;
  return base + __popcll(votes & before);
}

__device__ __forceinline__ void histogram_add(int* hist, bool valid, int digit) {
  // Concentrated waves issue one atomic, ordinary waves retain direct atomics.
  const int first = __shfl(digit, 0, 64);
  if (__all(valid && digit == first)) {
    if (threadIdx.x % 64 == 0) atomicAdd(hist + first, 64);
  } else if (valid) {
    atomicAdd(hist + digit, 1);
  }
  return;
}

// Full-width fallback or exact refinement of a certified, bounded candidate set.
// When selecting_id, select the smallest logical IDs among equal-score ties.
template <bool Cached, bool SelectingID, int K>
__device__ uint32_t refine(const float* input, int n, Shared<K>& s,
                          int rank, uint32_t score_boundary = 0) {
  const int tid = threadIdx.x;
  if (tid == 0) { s.prefix = 0; s.mask = 0; s.rank = rank; }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tid < 256) s.hist[tid] = 0;
    __syncthreads();
    const uint32_t prefix = s.prefix, mask = s.mask;
    for (int base = 0; base < n; base += Threads) {
      const int i = base + tid;
      uint32_t score_key = 0, candidate_key = 0;
      bool valid = i < n;
      if (valid) {
        score_key = Cached ? s.keys[i] : key(input[i]);
        const int id = Cached ? s.ids[i] : i;
        candidate_key = SelectingID ? ~static_cast<uint32_t>(id) : score_key;
        if (SelectingID) valid = score_key == score_boundary;
        valid = valid && ((candidate_key & mask) == prefix);
      }
      histogram_add(s.hist, valid, (candidate_key >> shift) & 255);
    }
    __syncthreads();
    int cumulative = 0, total = 0;
    const int count = tid < 256 ? s.hist[tid] : 0;
    Scan(s.temp.scan).InclusiveSum(count, cumulative, total);
    __syncthreads();
    const int above = total - cumulative;
    if (tid < 256 && above < s.rank && above + count >= s.rank) {
      s.active = count;
      s.hist[0] = s.rank - above;
      s.hist[1] = tid;
    }
    __syncthreads();
    if (tid == 0) {
      s.rank = s.hist[0];
      s.prefix |= static_cast<uint32_t>(s.hist[1]) << shift;
      s.mask |= 0xffu << shift;
    }
    __syncthreads();
    // The entire boundary bucket is needed; lower bits cannot affect selection.
    if (s.active == s.rank) break;
  }
  const uint32_t result = s.prefix;
  // A following tie refinement reuses prefix/rank. Every lane must read this
  // result before any lane can initialize the next refinement's shared state.
  __syncthreads();
  return result;
}

template <bool Cached, int K>
__device__ void exact_select(const float* input, int n, Shared<K>& s,
                             int requested = K, int output_offset = 0) {
  const int tid = threadIdx.x;
  const uint32_t boundary = refine<Cached, false>(input, n, s, requested);
  const int ties_needed = s.rank;
  const int ties_present = s.active;
  __syncthreads();
  uint32_t min_id_key = 0;
  if (ties_present > ties_needed) {
    min_id_key = refine<Cached, true>(input, n, s, ties_needed, boundary);
  }
  if (tid == 0) s.emitted = output_offset;
  __syncthreads();
  for (int base = 0; base < n; base += Threads) {
    const int i = base + tid;
    uint32_t value = 0;
    int id = 0;
    if (i < n) {
      value = Cached ? s.keys[i] : key(input[i]);
      id = Cached ? s.ids[i] : i;
    }
    const bool keep = i < n && (value > boundary ||
        (value == boundary && ~static_cast<uint32_t>(id) >= min_id_key));
    const int out = reserve_wave(keep, &s.emitted);
    if (keep && out < K) s.selected[out] = id;
  }
  __syncthreads();
  return;
}

__device__ __forceinline__ int coarse_bin(float value) {
  const uint16_t bits = __half_as_ushort(__float2half_rn(value));
  const uint16_t ordered = bits & 0x8000u ? static_cast<uint16_t>(~bits)
                                        : static_cast<uint16_t>(bits | 0x8000u);
  return ordered >> 8;
}

// Continue an oversized coarse bucket without restarting selection over all
// FP32 bits. Elements above the original bucket are already in selected[].
// Uncompacted rounds still read the row, but only the unresolved range votes.
template <int K>
__device__ void range_select(const float* input, int n, Shared<K>& s,
                             uint32_t low, uint32_t high, int remaining) {
  const int tid = threadIdx.x;
  const uint32_t original_high = high;
  const int offset = K-remaining;
  int rank = remaining;
  for (;;) {
    if (low == high) {
      // Exact-key plateau: retain higher keys and the earliest required IDs.
      if (tid == 0) { s.emitted = offset; s.eq_used = 0; }
      __syncthreads();
      for (int base=0; base<n; base+=Threads) {
        const int i=base+tid;
        const uint32_t v=i<n ? key(input[i]) : 0;
        const bool greater=i<n && v>low && v<=original_high;
        const int pos=reserve_wave(greater,&s.emitted);
        if (greater) s.selected[pos]=i;
        if (s.eq_used < rank) {
          const int match=i<n && v==low;
          int index=0,total=0;
          Scan(s.temp.scan).ExclusiveSum(match,index,total);
          __syncthreads();
          const int dest=K-rank+s.eq_used+index;
          if (match && dest<K) s.selected[dest]=i;
          __syncthreads();
          if (tid==0) s.eq_used+=total;
          __syncthreads();
        }
      }
      __syncthreads();
      return;
    }
    // Skip the bits shared by the entire certified range. This operates on
    // integer keys, preserving FP32 ordering without arithmetic rounding.
    const int shift=::max(0,32-__clz(low^high)-8);
    if (tid<256) s.hist[tid]=0;
    __syncthreads();
    for (int base=0; base<n; base+=Threads) {
      const int i=base+tid;
      const uint32_t v=i<n ? key(input[i]) : 0;
      histogram_add(s.hist,i<n && v>=low && v<=high,(v>>shift)&255);
    }
    __syncthreads();
    const int count=tid<256 ? s.hist[tid] : 0;
    int cumulative=0,total=0;
    Scan(s.temp.scan).InclusiveSum(count,cumulative,total);
    __syncthreads();
    const int above=total-cumulative;
    if (tid<256 && above<rank && above+count>=rank) {
      s.pivot=tid; s.rank=rank-above; s.active=count;
    }
    __syncthreads();
    const uint64_t digit_low=(static_cast<uint64_t>(low) &
        ~((uint64_t{1}<<(shift+8))-1)) | (static_cast<uint64_t>(s.pivot)<<shift);
    low=::max(low,static_cast<uint32_t>(digit_low));
    high=::min(high,static_cast<uint32_t>(digit_low+((uint64_t{1}<<shift)-1)));
    rank=s.rank;
    const int active=s.active;
    __syncthreads();
    if (active==rank) {
      if (tid==0) s.emitted=offset;
      __syncthreads();
      for (int base=0; base<n; base+=Threads) {
        const int i=base+tid;
        const uint32_t v=i<n ? key(input[i]) : 0;
        const bool keep=i<n && v>=low && v<=original_high;
        const int pos=reserve_wave(keep,&s.emitted);
        if (keep) s.selected[pos]=i;
      }
      __syncthreads();
      return;
    }
    if (active<=2*K) {
      if (tid==0) { s.emitted=offset; s.count=0; }
      __syncthreads();
      for (int base=0; base<n; base+=Threads) {
        const int i=base+tid;
        const uint32_t v=i<n ? key(input[i]) : 0;
        const bool greater=i<n && v>high && v<=original_high;
        const bool boundary=i<n && v>=low && v<=high;
        const int pos=reserve_wave(greater,&s.emitted);
        if (greater) s.selected[pos]=i;
        const int dest=reserve_wave(boundary,&s.count);
        if (boundary) { s.keys[dest]=v; s.ids[dest]=i; }
      }
      __syncthreads();
      exact_select<true>(input,active,s,rank,K-rank);
      return;
    }
  }
}

// Conventional exact path for shorter rows and rejected sampled proposals.
// The coarse FP16 conversion is only a monotone partition of non-NaN inputs;
// winners within the boundary bucket are always refined with full FP32 keys.
template <int K>
__device__ void coarse_select(const float* input, int n, Shared<K>& s) {
  constexpr int Capacity = 2*K;
  const int tid = threadIdx.x;
  if (tid < 256) s.hist[tid] = 0;
  if (tid == 0) s.count = 0;
  __syncthreads();
  Counts local{0, 0, 0};
  for (int base = 0; base < n; base += Threads) {
    const int i = base + tid;
    const float value = i < n ? input[i] : 0.f;
    const uint32_t bits = __float_as_uint(value);
    const uint32_t ordered = key(value);
    local.positive += i < n && ordered > 0x80000000u;
    local.zero += i < n && ordered == 0x80000000u;
    local.nan += i < n && (bits & 0x7fffffffu) > 0x7f800000u;
    // Positive-zero is counted by reduction, avoiding its hot histogram bin.
    histogram_add(s.hist, i < n && ordered != 0x80000000u, coarse_bin(value));
    const bool positive = i < n && ordered > 0x80000000u;
    const int dest = reserve_wave(positive, &s.count);
    if (positive && dest < K) s.ids[dest] = i;
  }
  __syncthreads();
  const Counts all = CountReduce(s.temp.counts).Reduce(local, AddCounts{});
  if (tid == 0) {
    s.count = all.positive; s.equal = all.zero; s.active = all.nan;
    s.hist[coarse_bin(0.f)] += all.zero;
  }
  __syncthreads();
  if (s.active) {
    exact_select<false>(input, n, s);
    return;
  }
  const int positive = s.count;
  if (positive <= K && positive + s.equal >= K) {
    // The first full scan already collected every positive ID. Only the
    // earliest zeros needed to fill K remain; no second full-row gather.
    for (int i=tid; i<positive; i+=Threads) s.selected[i]=s.ids[i];
    if (tid == 0) { s.emitted = 0; s.eq_used = 0; s.route = 7; }
    __syncthreads();
    for (int base = 0; base < n && s.eq_used < K-positive; base += Threads) {
      const int i = base + tid;
      const uint32_t value = i < n ? key(input[i]) : 0;
      if (s.eq_used < K-positive) {
        const int match = i < n && value == 0x80000000u;
        int pos = 0, total = 0;
        Scan(s.temp.scan).ExclusiveSum(match, pos, total);
        __syncthreads();
        const int out = positive + s.eq_used + pos;
        if (match && out < K) s.selected[out] = i;
        __syncthreads();
        if (tid == 0) s.eq_used += total;
        __syncthreads();
      }
    }
    __syncthreads();
    return;
  }
  int cumulative = 0, total = 0;
  const int count = tid < 256 ? s.hist[tid] : 0;
  Scan(s.temp.scan).InclusiveSum(count, cumulative, total);
  __syncthreads();
  const int above = total - cumulative;
  if (tid < 256 && above < K && above + count >= K) {
    s.pivot = tid; s.rank = K-above; s.active = count;
  }
  __syncthreads();
  const int bin = s.pivot, remaining = s.rank, boundary_size = s.active;
  __syncthreads();
  if (tid == 0) { s.count = 0; s.emitted = 0; }
  __syncthreads();
  KeyRange local_range{0xffffffffu,0};
  for (int base = 0; base < n; base += Threads) {
    const int i = base + tid;
    const float value = i < n ? input[i] : 0.f;
    const int digit = coarse_bin(value);
    const bool greater = i < n && digit > bin;
    const bool boundary = i < n && digit == bin;
    const int pos = reserve_wave(greater, &s.emitted);
    if (greater) s.selected[pos] = i;
    if (boundary_size <= Capacity) {
      const int dest = reserve_wave(boundary, &s.count);
      if (boundary) { s.keys[dest] = key(value); s.ids[dest] = i; }
    } else if (boundary) {
      const uint32_t v=key(value);
      local_range.low=::min(local_range.low,v);
      local_range.high=::max(local_range.high,v);
    }
  }
  __syncthreads();
  if (boundary_size > Capacity) {
    const KeyRange range=RangeReduce(s.temp.range).Reduce(local_range,MergeRange{});
    if (tid==0) { s.low=range.low; s.high=range.high; s.route=8; }
    __syncthreads();
    const uint32_t low=s.low,high=s.high;
    __syncthreads();
    range_select(input,n,s,low,high,remaining);
    return;
  }
  exact_select<true>(input, boundary_size, s, remaining, K-remaining);
  return;
}

} // namespace adaptive_topk_detail

template <int K, bool PhysicalPages = false>
__global__ __launch_bounds__(adaptive_topk_detail::Threads)
void adaptive_topk_kernel(const float* scores, int64_t score_stride, int width,
            const int* lengths, const int* starts, const int* pages,
            int64_t page_stride, int page_width, const int* cu, int batches,
            int* raw, int* mapped, int* stats, bool adaptive,
            uint32_t page_bits = 0, const int* offsets = nullptr) {
  using namespace adaptive_topk_detail;
  static_assert(K == 512 || K == 1024 || K == 2048);
  constexpr int Capacity = 2*K;
  __shared__ Shared<K> s;
  const int tid = threadIdx.x, row = blockIdx.x;
  const int start = starts ? starts[row] : 0;
  int n = lengths[row];
  if constexpr (PhysicalPages) {
    // Fixed Top512/1024 clamp lengths to score/page-table capacity.
    const int64_t capacity = ::min(static_cast<int64_t>(width),
                                 static_cast<int64_t>(page_width) << page_bits);
    n = static_cast<int>(::max(int64_t{0}, ::min(static_cast<int64_t>(n), capacity)));
  }
  const float* input = scores + row * score_stride + start;
  if (start < 0 || n < 0 || start > width || n > width - start) {
    for (int i = tid; i < K; i += Threads) {
      if (raw) raw[row*K+i] = -1;
      mapped[row*K+i] = -1;
    }
    if (tid == 0 && stats) stats[row*4] = -1;
    return;
  }
  if (tid == 0) { s.route = 0; s.count = 0; s.equal = 0; }
  __syncthreads();
  if (n <= K) {
    for (int i = tid; i < K; i += Threads) s.selected[i] = i < n ? i : INT_MAX;
    if (tid == 0) s.route = 1;
    __syncthreads();
  } else {
    bool resolved = false;
    if (adaptive && n >= 32768) {
      uint32_t samples[SampleItems];
      for (int j = 0; j < SampleItems; ++j) {
        const int sample_id = tid * SampleItems + j;
        samples[j] = key(input[static_cast<int64_t>(sample_id) * n / Samples]);
      }
      SampleSort(s.sample).SortDescending(samples);
      const int proposal_rank = ::min(Samples, ::max(1,
          static_cast<int>((int64_t{K+K/2}*Samples + n-1) / n)));
      for (int j = 0; j < SampleItems; ++j) {
        if (tid*SampleItems+j == proposal_rank-1) s.pivot = samples[j];
      }
      __syncthreads();
      const uint32_t pivot = s.pivot;
      int equal = 0;
      // Certification and candidate formation share the mandatory complete scan.
      for (int base = 0; base < n; base += Threads) {
        const int i = base + tid;
        const uint32_t value = i < n ? key(input[i]) : 0;
        const bool greater = i < n && value > pivot;
        equal += i < n && value == pivot;
        const int dest = reserve_wave(greater, &s.count);
        if (greater && dest < Capacity) { s.keys[dest] = value; s.ids[dest] = i; }
      }
      __syncthreads();
      const int equal_total = Reduce(s.temp.reduce).Sum(equal);
      if (tid == 0) s.equal = equal_total;
      __syncthreads();
      const int greater = s.count;
      if (greater <= K && greater + s.equal >= K) {
        // The pivot itself is the exact boundary, including large zero plateaus.
        for (int i = tid; i < greater; i += Threads) s.selected[i] = s.ids[i];
        if (tid == 0) { s.eq_used = 0; s.route = 2; }
        __syncthreads();
        for (int base = 0; base < n && s.eq_used < K-greater; base += Threads) {
          const int i = base + tid;
          const int match = i < n && key(input[i]) == pivot;
          int pos = 0, total = 0;
          Scan(s.temp.scan).ExclusiveSum(match, pos, total);
          __syncthreads();
          const int out = greater + s.eq_used + pos;
          if (match && out < K) s.selected[out] = i;
          __syncthreads();
          if (tid == 0) s.eq_used += total;
          __syncthreads();
        }
        resolved = true;
      } else if (greater >= K && greater <= Capacity) {
        if (tid == 0) s.route = 3;
        exact_select<true>(input, greater, s);
        resolved = true;
      } else {
        // No output is committed from truncated/underfilled candidates.
        if (tid == 0) s.route = greater > Capacity ? 4 : 5;
      }
      __syncthreads();
    }
    if (!resolved) {
      const int original_count = s.count, original_equal = s.equal;
      const int original_route = s.route;
      __syncthreads();
      if (tid == 0 && s.route == 0) s.route = 6;
      __syncthreads();
      coarse_select(input, n, s);
      if (tid == 0 && (original_route == 4 || original_route == 5)) {
        s.route = original_route; s.count = original_count; s.equal = original_equal;
      }
    }
    // The selected set and tie policy are exact; output order is unspecified.
    __syncthreads();
  }
  int batch = 0;
  // cu is small; only lane 0 resolves mapping, then publishes it to the block.
  if (tid == 0) {
    int low = row;
    if (cu) {
      low = 0;
      int high = batches;
      while (low < high) {
        int mid = (low + high) / 2;
        if (cu[mid+1] <= row) low = mid+1; else high = mid;
      }
    }
    s.eq_used = low;
  }
  __syncthreads();
  batch = s.eq_used;
  for (int i = tid; i < K; i += Threads) {
    const int id = s.selected[i];
    const bool valid = id >= 0 && id < n;
    if (raw) raw[row*K+i] = valid ? id : -1;
    int result = -1;
    if (valid) {
      if (offsets) {
        result = id + offsets[row];
      } else if (batch < batches) {
        const uint32_t slot = static_cast<uint32_t>(id) >> page_bits;
        if (slot < static_cast<uint32_t>(page_width)) {
          const int page = pages[batch*page_stride+slot];
          if constexpr (PhysicalPages) {
            if (page >= 0) result = (static_cast<uint32_t>(page) << page_bits) |
                (static_cast<uint32_t>(id) & ((uint32_t{1} << page_bits)-1));
          } else {
            result = page;
          }
        }
      }
    }
    mapped[row*K+i] = result;
  }
  if (tid == 0 && stats) {
    stats[row*4] = s.route;
    stats[row*4+1] = s.count;
    stats[row*4+2] = s.equal;
    stats[row*4+3] = n;
  }
}
} // namespace native
} // namespace at
