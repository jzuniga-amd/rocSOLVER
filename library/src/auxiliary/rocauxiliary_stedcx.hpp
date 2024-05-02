/************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_stebz.hpp"
#include "auxiliary/rocauxiliary_stein.hpp"
#include "lapack_device_functions.hpp"
#include "rocauxiliary_stedc.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

/***************** Device auxiliary functions *****************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDC_NUM_LEVELS returns the ideal number of times/levels in which a matrix (or split block)
    will be divided during the divide phase of divide & conquer algorithm.
    i.e. number of sub-blocks = 2^levels **/
template <>
__host__ __device__ inline rocblas_int
    stedc_num_levels<rocsolver_stedc_mode_bisection>(const rocblas_int n)
{
    rocblas_int levels = 0;
    // return the max number of levels such that the sub-blocks are at least of size 1
    // (i.e. 2^levels <= n), and there are no more than 256 sub-blocks (i.e. 2^levels <= 256)
    if(n <= 2)
        return levels;

    // TODO: tuning will be required; using the same tuning as QR for now
    if(n <= 4)
    {
        levels = 1;
    }
    else if(n <= 32)
    {
        levels = 2;
    }
    else if(n <= 232)
    {
        levels = 4;
    }
    else
    {
        if(n <= 1946)
        {
            if(n <= 1692)
            {
                if(n <= 295)
                {
                    levels = 5;
                }
                else
                {
                    levels = 7;
                }
            }
            else
            {
                levels = 7;
            }
        }
        else
        {
            levels = 8;
        }
    }

    return levels;
}

/*************** Main kernels *********************************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDCX_SPLIT_KERNEL implements the solver phase of the DC algorithm to
    **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEBZ_SPLIT_THDS)
    stedcx_split_kernel(const rocblas_erange range,
                        const rocblas_int n,
                        const S vl,
                        const S vu,
                        const rocblas_int il,
                        const rocblas_int iu,
                        S* DD,
                        const rocblas_stride strideD,
                        S* EE,
                        const rocblas_stride strideE,
                        S* WW,
                        const rocblas_stride strideW,
                        rocblas_int* splitsA,
                        S* workA,
                        const S eps,
                        const S ssfmin)
{
    // batch instance
    const int tid = hipThreadIdx_x;
    const int bid = hipBlockIdx_y;
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // workspace
    rocblas_int* ninter = splits + n + 2;
    rocblas_int* tmpIS = ninter + 2 * n;
    // using W as temp array to store the spit off-diagonal
    // (to use in case range = index)
    S* W = WW + bid * strideW;
    //nsplit is not needed; the number of split blocks goes into last entry
    //of splits when compact = true
    bool compact = true;
    rocblas_int* nsplit = nullptr;
    // range bounds
    S* bounds = workA + bid * (3 * n + 2);
    S* pivmin = bounds + 2;
    S* Esqr = pivmin + 1;
    S* inter = Esqr + n - 1;

    /*    T* pivmin = pivminA + bid;
    T* Esqr = EsqrA + bid * (n - 1);
    T* bounds = boundsA + 2 * bid;
    rocblas_int* tmpIS = tmpISA + (bid * n);
    T* inter = interA + bid * (2 * n);
    rocblas_int* ninter = ninterA + bid * (2 * n);
*/

    // shared memory setup for iamax.
    // (sidx also temporarily stores the number of blocks found by each thread)
    __shared__ S sval[STEBZ_SPLIT_THDS];
    __shared__ rocblas_int sidx[STEBZ_SPLIT_THDS];

    run_stebz_splitting<STEBZ_SPLIT_THDS>(tid, range, n, vl, vu, il, iu, D, E, nsplit, W, splits,
                                          tmpIS, pivmin, Esqr, bounds, inter, ninter, sval, sidx,
                                          eps, ssfmin, compact);
}

//--------------------------------------------------------------------------------------//
/** STEDCX_SOLVE_KERNEL implements the solver phase of the DC algorithm to
	compute the eigenvalues/eigenvectors of the different sub-blocks of each split-block.
	A matrix in the batch could have many split-blocks, and each split-block could be
	divided in a maximum of nn sub-blocks.
	- Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y
	  and nn groups in x. Groups are size STEDC_BDIM.
	- STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will be analysed
	  in parallel). If there are actually more split-blocks, some groups will work with more
	  than one split-block sequentially.
	- An upper bound for the number of sub-blocks (nn) can be estimated from the size n.
	  If a group has an id larger than the actual number of sub-blocks in a split-block,
	  it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedcx_solve_kernel(const rocblas_erange range,
                        const rocblas_int n,
                        const S vl,
                        const S vu,
                        const rocblas_int il,
                        const rocblas_int iu,
                        S* DD,
                        const rocblas_stride strideD,
                        S* EE,
                        const rocblas_stride strideE,
                        rocblas_int* nevA,
                        S* VA,
                        S* CC,
                        const rocblas_int shiftC,
                        const rocblas_int ldc,
                        const rocblas_stride strideC,
                        rocblas_int* iinfo,
                        S* WA,
                        rocblas_int* splitsA,
                        const S eps,
                        const S ssfmin,
                        const S ssfmax)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split-block id
    rocblas_int sid = hipBlockIdx_y;
    // sub-block id
    rocblas_int tid = hipBlockIdx_x;
    // thread index
    rocblas_int tidb = hipThreadIdx_x;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* info = iinfo + bid;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // workspace for solvers
    ////////////////    S* W = WA + bid * (2 + n * n);
    /* --------------------------------------------------- */

    // temporary arrays in shared memory
    /* --------------------------------------------------- */
    /*    extern __shared__ rocblas_int lsmem[];
    rocblas_int* sj2 = lsmem;
    S* sj1 = reinterpret_cast<S*>(sj2 + n + n % 2);*/
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // size of sub block
    rocblas_int sbs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        levs = stedc_num_levels<rocsolver_stedc_mode_jacobi>(bs);
        blks = 1 << levs;

        // 2. SOLVE PHASE
        /* ----------------------------------------------------------------- */
        // Solve the blks sub-blocks in parallel.

        /*        if(tid < blks)
        {
            sbs = ns[tid];
            p2 = ps[tid];

            // transform D and E into full upper tridiag matrix and copy to C
            de2tridiag(STEDC_BDIM, tidb, sbs, D + p2, E + p2, C + p2 + p2 * ldc, ldc);

            // set work space
            S* W_Acpy = W;
            S* W_residual = W_Acpy + n * n;
            rocblas_int* W_n_sweeps = reinterpret_cast<rocblas_int*>(W_residual + 1);

            // set shared mem
            rocblas_int even_n = sbs + sbs % 2;
            rocblas_int half_n = even_n / 2;
            S* cosines_res = sj1;
            S* sines_diag = cosines_res + half_n;
            rocblas_int* top = sj2;
            rocblas_int* bottom = top + half_n;

            // re-arrange threads in 2D array
            rocblas_int ddx, ddy;
            syevj_get_dims(sbs, STEDC_BDIM, &ddx, &ddy);
            rocblas_int tix = tidb % ddx;
            rocblas_int tiy = tidb / ddx;
            __syncthreads();

            // solve
            run_syevj<S, S>(ddx, ddy, tix, tiy, rocblas_esort_ascending, rocblas_evect_original,
                            rocblas_fill_upper, sbs, C + p2 + p2 * ldc, ldc, 0, eps, W_residual,
                            MAXSWEEPS, W_n_sweeps, D + p2, info, W_Acpy + p2 + p2 * n, cosines_res,
                            sines_diag, top, bottom);
            __syncthreads();
        }*/
    }
}

/******************* Host functions ********************************************/
/*******************************************************************************/

//--------------------------------------------------------------------------------------//
/** This helper calculates required workspace size **/
template <bool BATCHED, typename T, typename S>
void rocsolver_stedcx_getMemorySize(const rocblas_int n,
                                    const rocblas_int batch_count,
                                    size_t* size_work_stack,
                                    size_t* size_tempvect,
                                    size_t* size_tempgemm,
                                    size_t* size_tmpz,
                                    size_t* size_splits,
                                    size_t* size_workArr)
{
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    // if quick return no workspace needed
    if(n <= 1 || !batch_count)
    {
        *size_work_stack = 0;
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits = 0;
        *size_tmpz = 0;
        return;
    }

    size_t s1, s2, sq;

    // requirements for solver of small independent blocks
    rocsolver_steqr_getMemorySize<T, S>(rocblas_evect_tridiagonal, n, batch_count, &sq);
    s1 = sizeof(S) * (n + 2) * batch_count + std::max(sq, sizeof(S) * n * 2 * batch_count);

    // extra requirements for original eigenvectors of small independent blocks
    *size_tempvect = (n * n) * batch_count * sizeof(S);
    *size_tempgemm = 2 * (n * n) * batch_count * sizeof(S);
    if(COMPLEX)
        s2 = n * n * batch_count * sizeof(S);
    else
        s2 = 0;
    if(BATCHED && !COMPLEX)
        *size_workArr = sizeof(S*) * batch_count;
    else
        *size_workArr = 0;
    *size_work_stack = std::max(s1, s2);

    // size for split blocks and sub-blocks positions
    *size_splits = sizeof(rocblas_int) * (5 * n + 2) * batch_count;

    // size for temporary diagonal and rank-1 modif vector
    *size_tmpz = sizeof(S) * (2 * n) * batch_count;
}

//--------------------------------------------------------------------------------------//
/** Helper to check argument correctnesss **/
template <typename T, typename S>
rocblas_status rocsolver_stedcx_argCheck(rocblas_handle handle,
                                         const rocblas_erange range,
                                         const rocblas_int n,
                                         const S vlow,
                                         const S vup,
                                         const rocblas_int ilow,
                                         const rocblas_int iup,
                                         S* D,
                                         S* E,
                                         rocblas_int* nev,
                                         S* W,
                                         T* C,
                                         const rocblas_int ldc,
                                         rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(range != rocblas_erange_all && range != rocblas_erange_value && range != rocblas_erange_index)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || ldc < n)
        return rocblas_status_invalid_size;
    if(range == rocblas_erange_value && vlow >= vup)
        return rocblas_status_invalid_size;
    if(range == rocblas_erange_index && (iup > n || (n > 0 && ilow > iup)))
        return rocblas_status_invalid_size;
    if(range == rocblas_erange_index && (ilow < 1 || iup < 0))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && (!D || !W || !C)) || (n > 1 && !E) || !info || !nev)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//--------------------------------------------------------------------------------------//
/** STEDCX templated function **/
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_stedcx_template(rocblas_handle handle,
                                         const rocblas_erange erange,
                                         const rocblas_int n,
                                         const S vl,
                                         const S vu,
                                         const rocblas_int il,
                                         const rocblas_int iu,
                                         S* D,
                                         const rocblas_stride strideD,
                                         S* E,
                                         const rocblas_stride strideE,
                                         rocblas_int* nev,
                                         S* W,
                                         const rocblas_stride strideW,
                                         U C,
                                         const rocblas_int shiftC,
                                         const rocblas_int ldc,
                                         const rocblas_stride strideC,
                                         rocblas_int* info,
                                         const rocblas_int batch_count,
                                         S* work_stack,
                                         S* tempvect,
                                         S* tempgemm,
                                         S* tmpz,
                                         rocblas_int* splits,
                                         S** workArr)
{
    ROCSOLVER_ENTER("stedcx", "erange:", erange, "n:", n, "vl:", vl, "vu:", vu, "il:", il,
                    "iu:", iu, "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 1)
        ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0, stream, C,
                                strideC, n, 1);
    if(n <= 1)
        return rocblas_status_success;

    // constants
    S eps = get_epsilon<S>();
    S ssfmin = get_safemin<S>();
    S ssfmax = S(1.0) / ssfmin;
    ssfmin = sqrt(ssfmin) / (eps * eps);
    ssfmax = sqrt(ssfmax) / S(3.0);
    rocblas_int blocksn = (n - 1) / BS2 + 1;

    //print_device_matrix(std::cout,"D",1,n,D,1);
    //print_device_matrix(std::cout,"E",1,n-1,E,1);

    // initialize identity matrix in C if required
    ROCSOLVER_LAUNCH_KERNEL(init_ident<T>, dim3(blocksn, blocksn, batch_count), dim3(BS2, BS2), 0,
                            stream, n, n, C, shiftC, ldc, strideC);

    // initialize identity matrix in tempvect
    rocblas_int ldt = n;
    rocblas_stride strideT = n * n;
    ROCSOLVER_LAUNCH_KERNEL(init_ident<S>, dim3(blocksn, blocksn, batch_count), dim3(BS2, BS2), 0,
                            stream, n, n, tempvect, 0, ldt, strideT);

    // find max number of sub-blocks to consider during the divide phase
    rocblas_int maxlevs = stedc_num_levels<rocsolver_stedc_mode_bisection>(n);
    rocblas_int maxblks = 1 << maxlevs;

    // find independent split blocks in matrix and prepare range for partial decomposition
    ROCSOLVER_LAUNCH_KERNEL(stedcx_split_kernel, dim3(1, batch_count), dim3(STEBZ_SPLIT_THDS), 0,
                            stream, erange, n, vl, vu, il, iu, D, strideD, E, strideE, W, strideW,
                            splits, work_stack, eps, ssfmin);
    //printf("after splits\n");
    //print_device_matrix(std::cout,"D",1,n,D,1);
    //print_device_matrix(std::cout,"E",1,n-1,E,1);
    //print_device_matrix(std::cout,"splits",1,5*n+2,splits,1);

    // 1. divide phase
    //-----------------------------
    ROCSOLVER_LAUNCH_KERNEL((stedc_divide_kernel<rocsolver_stedc_mode_bisection, S>),
                            dim3(batch_count), dim3(STEDC_BDIM), 0, stream, n, D, strideD, E,
                            strideE, splits);
    //printf("after divide\n");
    //print_device_matrix(std::cout,"D",1,n,D,1);
    //print_device_matrix(std::cout,"E",1,n-1,E,1);
    //print_device_matrix(std::cout,"splits",1,5*n+2,splits,1);

    // 2. solve phase
    //-----------------------------
    ROCSOLVER_LAUNCH_KERNEL((stedc_solve_kernel<S>), dim3(maxblks, STEDC_NUM_SPLIT_BLKS, batch_count),
                            dim3(1), 0, stream, n, D, strideD, E, strideE, tempvect, 0, ldt,
                            strideT, info, work_stack + n + 2, splits, eps, ssfmin, ssfmax);
    //printf("after solve\n");
    //print_device_matrix(std::cout,"D",1,n,D,1);
    //print_device_matrix(std::cout,"E",1,n-1,E,1);
    //print_device_matrix(std::cout,"splits",1,5*n+2,splits,1);
    //print_device_matrix(std::cout,"C",n,n,tempvect,ldt);

    // 3. merge phase
    //----------------
    size_t lmemsize = sizeof(S) * STEDC_BDIM;
    for(rocblas_int k = 0; k < maxlevs; ++k)
    {
        // at level k numgrps thread-groups are needed
        rocblas_int numgrps = 1 << (maxlevs - 1 - k);

        // launch merge for level k
        /*        ROCSOLVER_LAUNCH_KERNEL((stedc_merge_kernel<rocsolver_stedc_mode_bisection, S>),
                            dim3(numgrps, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(STEDC_BDIM), lmemsize,
                            stream, k, n, D, strideD, E, strideE, tempvect, 0, ldt, strideT, tmpz,
                            tempgemm, splits, eps, ssfmin, ssfmax);*/
    }

    // 4. update and sort
    //----------------------
    // eigenvectors C <- C*tempvect
    local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                    work_stack, 0, ldt, strideT, batch_count, workArr);

    ROCSOLVER_LAUNCH_KERNEL((stedc_sort<T>), dim3(batch_count), dim3(1), 0, stream, n, D, strideD,
                            C, shiftC, ldc, strideC);

    //print_device_matrix(std::cout,"W",1,n,D,1);
    //print_device_matrix(std::cout,"C",n,n,C,ldc);

    return rocblas_status_success;
}
