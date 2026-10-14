// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef LIGHTOP_LMSLIMQUANT_NATIVE_BASE_BASE_CUH_
#define LIGHTOP_LMSLIMQUANT_NATIVE_BASE_BASE_CUH_
#define BLOCKSIZE 256

#include <iostream>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <string>
#include <cstdlib>
#include <sys/time.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#pragma once
template <typename T>
void PrintScale(hipStream_t stream,const T* scales_and_zeros,int size);

struct my_timer
{
    timeval ts, te; //起始时刻，终止时刻
    float dt; // 时间间隔,单位毫秒(ms)
    void start(){
        gettimeofday(&ts, NULL);
    }
    void stop(){
        gettimeofday(&te, NULL);
        long int dt_sec  = te.tv_sec - ts.tv_sec;
        long int dt_usec = te.tv_usec - ts.tv_usec;
        dt = dt_sec * 1.0e3 + dt_usec / 1.0e3;
    }
};
#endif

template<typename T, int N>
struct Array {

    using value_type      = T;
    using size_type       = int;
    using difference_type = int;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    static_assert(N > 0);

    T __a[N];

    __device__ __host__ constexpr reference operator[](size_type i) noexcept
    {
        return __a[i];
    }
    __device__ __host__ constexpr const_reference operator[](size_type i) const noexcept
    {
        return __a[i];
    }

    __device__ __host__ constexpr reference front() noexcept
    {
        return *begin();
    }

    __device__ __host__ constexpr const_reference front() const noexcept
    {
        return *begin();
    }

    __device__ __host__ constexpr reference back() noexcept
    {
        return *(end() - 1);
    }

    __device__ __host__ constexpr const_reference back() const noexcept
    {
        return *(end() - 1);
    }

    __device__ __host__ constexpr pointer data() noexcept
    {
        return &__a[0];
    }

    __device__ __host__ constexpr const_pointer data() const noexcept
    {
        return &__a[0];
    }

    __device__ __host__ constexpr iterator begin() noexcept
    {
        return data();
    }

    __device__ __host__ constexpr const_iterator begin() const noexcept
    {
        return data();
    }

    __device__ __host__ constexpr iterator end() noexcept
    {
        return data() + N;
    }

    __device__ __host__ constexpr const_iterator end() const noexcept
    {
        return data() + N;
    }

    __device__ __host__ constexpr std::integral_constant<int, N> size() const noexcept
    {
        return {};
    }

    __device__ __host__ constexpr std::false_type empty() const noexcept
    {
        return {};
    }
};
