/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "hip/amd_detail/amd_warp_sync_functions.h"
#include "hip/device_functions.h"
#include "hip/hip_common.h"
#include "hip/hip_cooperative_groups.h"

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

ROCSOLVER_BEGIN_NAMESPACE

#ifndef HIP_CHECK
#define HIP_CHECK(fcn)               \
    {                                \
        auto istat = (fcn);          \
        assert(istat == hipSuccess); \
    }
#endif

#ifndef LAUNCH_CHECK
#define LAUNCH_CHECK(fcn)                                                                  \
    {                                                                                      \
        auto const istat = (fcn);                                                          \
        bool const isok = (istat == hipSuccess);                                           \
        if(!isok)                                                                          \
        {                                                                                  \
            std::cerr << "Kernel launch error: " << hipGetErrorString(istat) << std::endl; \
        }                                                                                  \
        assert(isok);                                                                      \
    }
#endif

namespace cg = cooperative_groups;

template <typename T, typename I>
__device__ T reduce_sum_shfl_wsize(I const wsize, T val)
{
    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    if(wsize == 64)
    {
        val += __shfl_down(val, 32); // offset = 32
        val += __shfl_down(val, 16); // offset = 16
        val += __shfl_down(val, 8); // offset = 8
        val += __shfl_down(val, 4); // offset = 4
        val += __shfl_down(val, 2); // offset = 2
        val += __shfl_down(val, 1); // offset = 1
    }
    else if(wsize == 32)
    {
        val += __shfl_down(val, 16); // offset = 16
        val += __shfl_down(val, 8); // offset = 8
        val += __shfl_down(val, 4); // offset = 4
        val += __shfl_down(val, 2); // offset = 2
        val += __shfl_down(val, 1); // offset = 1
    }
    else
    {
        for(auto offset = wsize / 2; offset > 0; offset /= 2)
        {
            val += __shfl_down(val, offset);
            // g.sync();
        }
    }
    return val; // note: only thread 0 will return full sum
}

static bool get_cooperative_launch(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeCooperativeLaunch;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_lds_size(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMaxSharedMemoryPerBlock;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_num_cu(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMultiprocessorCount;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_warp_size(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeWarpSize;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_max_threads_per_block(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMaxThreadsPerBlock;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

// ---------------------------------------
// compute rank-1 update in a thread block
// A = alpha * x * y' + A
// ---------------------------------------
template <typename T, typename I>
__device__ void gerc_body(I const m,
                          I const n,
                          T const alpha,
                          T const* const x_,
                          I const incx,
                          T const* const y_,
                          I const incy,
                          T* const A_,
                          I const lda)

{
    bool const has_work = (m >= 1) && (n >= 1);
    if(!has_work)
    {
        return;
    };

    bool constexpr is_complex = rocblas_is_complex<T>;

    I const i_start = threadIdx.x;
    I const i_inc = blockDim.x;

    I const j_start = threadIdx.y;
    I const j_inc = blockDim.y;

    auto x = [=](auto i) { return ((incx == 1) ? x_[i] : x_[i * static_cast<int64_t>(incx)]); };

    auto y = [=](auto i) { return ((incy == 1) ? y_[i] : y_[i * static_cast<int64_t>(incy)]); };

    for(I j = j_start; j < n; j += j_inc)
    {
        T const yj = y(j);
        T conj_yj = yj;
        if constexpr(is_complex)
        {
            conj_yj = std::conj(yj);
        }

        auto const alpha_conj_yj = alpha * conj_yj;
        auto const joffset = j * static_cast<int64_t>(lda);
        for(I i = i_start; i < m; i += i_inc)
        {
            A_[i + joffset] += alpha_conj_yj * x(i);
        }
    }
}

// ------------------------
// compute  y = beta * y + alpha * A * x
// or
// y = beta * y +  alpha * A' * x
// ------------------------
template <typename T, typename I>
__device__ void gemv_body(char const trans,
                          I const m,
                          I const n,
                          T const alpha,
                          T const* const A_,
                          I const lda,
                          T const* const x_,
                          I const incx,
                          T const beta,
                          T* const y_,
                          I const incy)
{
    bool const has_work = (m >= 1) && (n >= 1);
    if(!has_work)
    {
        return;
    };

    I const itx = threadIdx.x;
    I const nx = blockDim.x;

    I const ity = threadIdx.y;
    I const ny = blockDim.y;

    bool const is_conj_transpose = (trans == 'C') || (trans == 'c');
    bool const is_transpose = (trans == 'T') || (trans == 't');
    bool const is_no_transpose = (!is_conj_transpose) && (!is_transpose);

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto A = [=](auto i, auto j) -> const T& { return (A_[idx2D(i, j, lda)]); };

    auto x = [=](auto i) -> const T& {
        return ((incx == 1) ? x_[i] : x_[i * static_cast<int64_t>(incx)]);
    };

    auto y = [=](auto i) -> T& { return ((incy == 1) ? y_[i] : y_[i * static_cast<int64_t>(incy)]); };

    bool const is_beta_zero = (beta == 0);

    bool const is_complex = rocblas_is_complex<T>;

    if(is_no_transpose)
    {
        I const i_start = ity;
        I const i_inc = ny;

        I const j_start = itx;
        I const j_inc = nx;

        // ------------------------------------------------------------------
        // y(0:(m-1)) = beta * y(0:(m-1)) + A(0:(m-1), 0:(n-1)) * x(0:(n-1))
        // ------------------------------------------------------------------
        for(I i = i_start; i < m; i += i_inc)
        {
            T yi = 0;

            for(I j = j_start; j < n; j += j_inc)
            {
                yi += A(i, j) * x(j);
            }

            // -----------------------------------------
            // note: only thread 0 has the correct value
            // after sum reduction
            // -----------------------------------------
            if(j_inc > 1)
            {
                yi = reduce_sum_shfl_wsize(j_inc, yi);
            }

            if(j_start == 0)
            {
                if(is_beta_zero)
                {
                    y(i) = alpha * yi;
                }
                else
                {
                    y(i) = beta * y(i) + alpha * yi;
                }
            }
        }
    }
    else
    {
        // -----------------------------------------------------------------------
        // y(0:(n-1)) = beta * y(0:(n-1)) * trans( A(0:(m-1),0:(n-1)) * x(0:(m-1))
        // -----------------------------------------------------------------------

        I j_start = ity;
        I j_inc = ny;

        I i_start = itx;
        I i_inc = nx;

        for(I j = j_start; j < n; j += j_inc)
        {
            T yj = 0;
            for(I i = i_start; i < m; i += i_inc)
            {
                T const aij = A(i, j);
                if constexpr(is_complex)
                {
                    auto const atji = (is_conj_transpose) ? std::conj(aij) : aij;
                    yj += atji * x(i);
                }
                else
                {
                    yj += aij * x(i);
                }
            };

            // -----------------------------------------
            // note: only thread 0 has the correct value
            // after sum reduction
            // -----------------------------------------
            if(i_inc > 1)
            {
                yj = reduce_sum_shfl_wsize(i_inc, yj);
            }

            if(i_start == 0)
            {
                if(is_beta_zero)
                {
                    y(j) = alpha * yj;
                }
                else
                {
                    y(j) = beta * y(j) + alpha * yj;
                }
            }
        }
    }
}

//  ------------------------------------------------------------------
//  compute  C( 0:(m-1), 0:(n-1)) <- H * C( 0:(m-1), 0:(n-1)),
//  where H = eye - tau * v * v'
//  or
//  C(0:(m-1),0:(n-1))  <- C(0:(m-1), 0:(n-1)) * H( 0:(n-1), 0:(n-1) )
//
//  performed in a single thread block
//  ------------------------------------------------------------------
template <typename T, typename I>
__device__ void larf_body(char const side,
                          I const m,
                          I const n,
                          T const* const v_,
                          I const incv,
                          T const tau,
                          T* const C,
                          I const ldc,
                          T* const w,
                          I const wsize,
                          I const jb_start,
                          I const jb_inc)
{
    bool const is_applyleft = (side == 'L') || (side == 'l');

    T const one = 1;
    T const zero = 0;
    I const incw = 1;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    bool const is_complex = rocblas_is_complex<T>;

    if(is_applyleft)
    {
        // ----------------------------------------------------------------------
        // form C(0:(m-1),0:(n-1)) <- H( 0:(m-1), 0:(m-1)) * C( (0:(m-1), 0:(n-1))
        // where
        // H(0:(n-1),0:(n-1) = eye - tau * v( 0:(n-1)) * v( 0:(n-1))'
        // ----------------------------------------------------------------------

        //  ---------------------------------------
        // Note   C = (I - tau * v * v') * C
        //          = C - tau * v * ( v' * C)
        //
        //          let w' = v' * C, or w = C' * v
        //
        //  C = C - tau * v * w'
        // ------------------------------------------

        // ---------------------------------------------
        // store vector w  in LDS shared memory
        // wsize is max length of w vector that can fit
        // in LDS shared memory
        //
        //  [C1 | C2 ] = [C1 | C2] - tau * v * v' * [C1 | C2 ]
        //  [C1 | C2 ] = [C1 | C2] - tau * v * [ w1' | w2' ]
        //
        //  where w1' = v' * C1, or w1 = C1' * v
        //        w2' = v' * C2, or w2 = C2' * v
        //  ---------------------------------------

        for(I jb = jb_start; jb < ceil(n, wsize); jb += jb_inc)
        {
            I const j = jb * wsize;
            I const jend = std::min(n, j + wsize);
            I const mm = m;
            I const nn = (jend - j);

            bool const has_work = (mm >= 1) && (nn >= 1);
            if(!has_work)
            {
                continue;
            };

            {
                T const* const Cmat = C + idx2D(0, j, ldc);
                char const trans = (is_complex) ? 'C' : 'T';
                gemv_body<T, I>(trans, mm, nn, one, Cmat, ldc, v_, incv, zero, w, incw);
                __syncthreads();
            }

            {
                T* const Cmat = C + idx2D(0, j, ldc);
                auto const alpha = -tau;
                gerc_body<T, I>(mm, nn, alpha, v_, incv, w, incw, Cmat, ldc);
                __syncthreads();
            }
        }
    }
    else
    {
        // ----------------------------------------------------------------------
        // form C(0:(m-1),0:(n-1)) <- C( (0:(m-1), 0:(n-1)) * H( 0:(n-1), 0:(n-1))
        // where
        // H(0:(n-1),0:(n-1) = eye - tau * v( 0:(n-1)) * v( 0:(n-1))'
        // -----------------------------------------------------------------

        // -----------------------------------------------------
        // Note
        // [C1] = [C1] * H = [C1 * H]
        // [--]   [--]       [------]
        // [C2]   [C2]       [C2 * H]
        // -----------------------------------------------------

        for(I jb = jb_start; jb < ceil(m, wsize); jb += jb_inc)
        {
            I const j = jb * wsize;
            I const jend = std::min(m, j + wsize);

            I const mm = (jend - j);
            I const nn = n;

            {
                T const* const Cmat = C + idx2D(j, 0, ldc);
                char const trans = 'N';
                gemv_body<T, I>(trans, mm, nn, one, Cmat, ldc, v_, incv, zero, w, incw);
                __syncthreads();
            }

            {
                T* const Cmat = C + idx2D(j, 0, ldc);
                auto const alpha = -tau;
                gerc_body<T, I>(mm, nn, alpha, w, incw, v_, incv, Cmat, ldc);
                __syncthreads();
            }
        }
    }
}

// ------------------------------------------------------
// larf but v_[0] is assumed to be 1 and
// the memory location is used to hold tau
// ------------------------------------------------------
template <typename T, typename I>
__device__ void ularf_body(char const side,
                           I const m,
                           I const n,
                           T* const v_,
                           I const incv,
                           T* const C,
                           I const ldc,
                           size_t const ldsmax,
                           I const jb_start,
                           I const jb_inc)
{
    bool const is_applyleft = (side == 'L') || (side == 'l');

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    // -----------------------------------------------------
    // allocate  temporary vector "w" from LDS shared memory
    // -----------------------------------------------------
    extern __shared__ double ldsmem[];

    I const vlen = (is_applyleft) ? m : n;
    I const vsize = sizeof(T) * vlen;

    I const wlen_max = (ldsmax - vsize) / sizeof(T);
    assert(wlen_max >= 1);

    I const lenC = (is_applyleft) ? n : m;
    I const wlen = std::max(I{1}, std::min(wlen_max, ceil(lenC, jb_inc)));

    std::byte* pfree = (std::byte*)&(ldsmem[0]);
    T* const vsh = (T*)pfree;
    pfree += sizeof(T) * vlen;
    T* const w = (T*)pfree;
    pfree += sizeof(T) * wlen;

    I const tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    I const nthreads = (blockDim.x * blockDim.y) * blockDim.z;

    T tau = v_[0];

    auto const i_start = tid;
    auto const i_inc = nthreads;

    __syncthreads();
    for(auto i = i_start; i < vlen; i += i_inc)
    {
        vsh[i] = (i == 0) ? 1 : v_[i];
    }
    __syncthreads();

    larf_body(side, m, n, vsh, incv, tau, C, ldc, w, wlen, jb_start, jb_inc);
}

template <typename T, typename I, typename Istride, typename TA, typename TC>
__device__ void ormtr_sb2st_body(I const ksweep,
                                 I const n,
                                 I const noffdiag,

                                 TA A_,
                                 Istride const shiftA,
                                 I const lda,
                                 Istride strideA,

                                 TC C_,
                                 Istride const shiftC,
                                 I const ldc,
                                 Istride strideC,

                                 I const batch_count,
                                 size_t const ldsmax,

                                 I const bid_start,
                                 I const bid_inc,
                                 I const igroup_start,
                                 I const igroup_inc,
                                 I const jb_start,
                                 I const jb_inc)
{
    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };
    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto const A = load_ptr_batch(A_, bid, shiftA, strideA);
        auto const C = load_ptr_batch(C_, bid, shiftC, strideC);

        /*
     * -------------------------------------------
     * Encoding of Householder vectors in matrix A
     * with noffdiag == 3
     * -------------------------------------------

    [1          ]
    [t 1        ]
    [v t 1      ]
    [v v t 1    ]
    [t v v t 1  ]
    [v t v v t 1]

    */

        auto const vlen = (n - 1) - ksweep;
        auto const ngroups = ceil(vlen, noffdiag);
        auto group_size = [=](auto igroup) {
            auto const v_remain = (vlen - (ngroups - 1) * noffdiag);
            bool const is_last_group = (igroup == (ngroups - 1));
            return (is_last_group ? v_remain : noffdiag);
        };

        for(I igroup = igroup_start; igroup < ngroups; igroup += igroup_inc)
        {
            auto const ioffset_c = ksweep + 1;
            auto const irow_c = ioffset_c + igroup * noffdiag;
            auto const Cp = C + idx2D(irow_c, 0, ldc);

            char const side = 'L'; // left side
            I const mm = group_size(igroup);
            I const nn = n;
            T* const v_ = A + idx2D(irow_c, ksweep, lda);
            I const incv = 1;
            ularf_body(side, mm, nn, v_, incv, Cp, ldc, ldsmax, jb_start, jb_inc);
        }
    } // end for bid
}

// -------------------------------------------------------
// perform Householder transformations to C matrix
// related to reducing banded matrix to tridiagonal matrix
// -------------------------------------------------------
template <typename T, typename I, typename Istride, typename TA, typename TC>
static __global__ void ormtr_sb2st_coop_kernel(I const n,
                                               I const noffdiag,

                                               TA A_,
                                               Istride const shiftA,
                                               I const lda,
                                               Istride const strideA,

                                               TC C_,
                                               Istride const shiftC,
                                               I const ldc,
                                               Istride const strideC,

                                               I const batch_count,
                                               size_t const ldsmax)
{
    I const nblocks = (gridDim.x * gridDim.y) * gridDim.z;
    I const block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * (gridDim.x * gridDim.y);

    // -----------------------------------------------------
    // if batch_count is large, then assign one thread block
    // to each batch item
    // -----------------------------------------------------
    bool const is_large_batch_count = (batch_count >= nblocks);

    I const bid_start = (is_large_batch_count) ? block_id : 0;
    I const bid_inc = (is_large_batch_count) ? nblocks : 1;

    I nbx = 1;
    I nby = nblocks;

    // ---------------------------
    // partition the thread blocks
    // on nbx by nby  2D grid
    //
    // current block is entry (ibx,iby)
    // ---------------------------
    if((nblocks % 4) == 0)
    {
        nbx = 4;
        nby = nblocks / 4;
    }
    else if((nblocks % 2) == 0)
    {
        nbx = 2;
        nby = nblocks / 2;
    }

    I const ibx = block_id % nbx;
    I const iby = block_id / nbx;

    I const igroup_start = (is_large_batch_count) ? 0 : iby;
    I const igroup_inc = (is_large_batch_count) ? 1 : nby;

    I const jb_start = (is_large_batch_count) ? 0 : ibx;
    I const jb_inc = (is_large_batch_count) ? 1 : nbx;

    for(I ksweep = (n - 1) - 1; ksweep >= 0; ksweep--)
    {
        ormtr_sb2st_body<T, I, Istride, TA, TC>(ksweep, n, noffdiag,

                                                A_, shiftA, lda, strideA,

                                                C_, shiftC, ldc, strideC,

                                                batch_count, ldsmax,

                                                bid_start, bid_inc, igroup_start, igroup_inc,
                                                jb_start, jb_inc);
        // -------------------------------------
        // before performing transformation for next sweep,
        // synchronize the entire grid of blocks of threads
        // without re-launching kernel
        // -------------------------------------
        cg::this_grid().sync();
    }
}

// -------------------------------------------------------
// perform a single sweep of Householder transformations to C matrix
// related to reducing banded matrix to tridiagonal matrix
// -------------------------------------------------------
template <typename T, typename I, typename Istride, typename TA, typename TC>
static __global__ void ormtr_sb2st_kernel(I const ksweep,
                                          I const n,
                                          I const noffdiag,

                                          TA A_,
                                          Istride const shiftA,
                                          I const lda,
                                          Istride const strideA,

                                          TC C_,
                                          Istride const shiftC,
                                          I const ldc,
                                          Istride const strideC,

                                          I const batch_count,
                                          size_t const ldsmax = 64 * 1024)
{
    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const ibx = blockIdx.x;
    I const iby = blockIdx.y;
    I const nbx = gridDim.x;
    I const nby = gridDim.y;

    I const igroup_start = iby;
    I const igroup_inc = nby;

    I const jb_start = ibx;
    I const jb_inc = nbx;

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    ormtr_sb2st_body<T, I, Istride, TA, TC>(ksweep, n, noffdiag,

                                            A_, shiftA, lda, strideA,

                                            C_, shiftC, ldc, strideC,

                                            batch_count, ldsmax,

                                            bid_start, bid_inc, igroup_start, igroup_inc, jb_start,
                                            jb_inc);
}

template <typename T, typename I, typename Istride, typename TA, typename TC>
static void ormtr_sb2st_template(hipStream_t stream,

                                 I const n,
                                 I const noffdiag,

                                 TA A_,
                                 Istride const shiftA,
                                 I const lda,
                                 Istride const strideA,

                                 TC C_,
                                 Istride const shiftC,
                                 I const ldc,
                                 Istride const strideC,

                                 I const batch_count,
                                 bool const use_cooperative_kernel = false)
{
    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    // configure kernels
    I const num_cu = get_num_cu();
    I const warp_size = get_warp_size();
    I const max_threads_per_block = get_max_threads_per_block();
    I const lds_size = get_lds_size();

    I const nx = warp_size;
    I const ny = max_threads_per_block / nx;

    if(use_cooperative_kernel)
    {
        void* args[] = {(void*)&n,           (void*)&noffdiag,

                        (void*)&A_,          (void*)&shiftA,   (void*)&lda, (void*)&strideA,

                        (void*)&C_,          (void*)&shiftC,   (void*)&ldc, (void*)&strideC,

                        (void*)&batch_count, (void*)&lds_size};

        LAUNCH_CHECK(
            hipLaunchCooperativeKernel((void*)(ormtr_sb2st_coop_kernel<T, I, Istride, TA, TC>),
                                       dim3(num_cu, 1, 1), dim3(nx, ny, 1), args, lds_size, stream));
    }
    else
    {
        for(I ksweep = (n - 1) - 1; ksweep >= 0; ksweep--)
        {
            I const clen = (n - 1) - ksweep;
            I const wsize_max = (lds_size - noffdiag * sizeof(T)) / sizeof(T);
            I const wsize = std::min(wsize_max, 256);

            I const nbx = ceil(n, wsize);
            I const nby = ceil(clen, noffdiag);
            I const nbz = std::min(I{1024}, batch_count);

            ormtr_sb2st_kernel<T, I, Istride, TA, TC>
                <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), lds_size, stream>>>(ksweep, n, noffdiag,

                                                                             A_, shiftA, lda, strideA,

                                                                             C_, shiftC, ldc, strideC,

                                                                             batch_count, lds_size);
        }
    }
}

ROCSOLVER_END_NAMESPACE
