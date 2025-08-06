/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack_device_functions.hpp"
#include "rocauxiliary_steqr.hpp"
#include "rocauxiliary_sterf.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

#include <algorithm>

ROCSOLVER_BEGIN_NAMESPACE

#define STEDC_BDIM 512  // Number of threads per thread-block used in main stedc kernels
#define WAVEFRONT 64    // Number of threads in a wavefront

// TODO: using macro STEDC_EXTERNAL_GEMM = true for now. In the future we can pass
// STEDC_EXTERNAL_GEMM at run time to switch between internal vector updates and
// external gemm-based updates.
#define STEDC_EXTERNAL_GEMM true


/*************** Main kernels *********************************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDC_DIVIDE_KERNEL implements the divide phase of the DC algorithm. It
    divides the input matrix into 'blks' sub-blocks.
        - This kernel is to be called with as many groups in x as needed to cover all
        the batch_count problems. 
        - Each thread will work with a matrix in the batch. 
        - Size of groups is set to STEDC_BDIM. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) 
stedc_divide_kernel(const rocblas_int levs,
                    const rocblas_int blks,
                    const rocblas_int n,
                    S* DD,
                    const rocblas_stride strideD,
                    S* EE,
                    const rocblas_stride strideE,
                    const rocblas_int batch_count,
                    rocblas_int* splitsA)
{
    // threads and groups indices
    rocblas_int bid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // for each matrix in the batch
    if(bid < batch_count)
    {
        // select batch instance to work with
        S* D = DD + bid * strideD;
        S* E = EE + bid * strideE;

        // temporary arrays in global memory
        rocblas_int* splits = splitsA + bid * (5 * n + 2);
        // the sub-blocks sizes
        rocblas_int* ns = splits + n + 2;
        // the sub-blocks initial positions
        rocblas_int* ps = ns + n;

        // find sizes of sub-blocks
        ns[0] = n;
        rocblas_int t, t2;
        for(int i = 0; i < levs; ++i)
        {
            for(int j = (1 << i); j > 0; --j)
            {
                t = ns[j - 1];
                t2 = t / 2;
                ns[j * 2 - 1] = (2 * t2 < t) ? t2 + 1 : t2;
                ns[j * 2 - 2] = t2;
            }
        }

        // find beginning of sub-blocks and update elements in D
        rocblas_int p2 = 0;
        ps[0] = p2;
        for(int i = 1; i < blks; ++i)
        {
            p2 += ns[i - 1];
            ps[i] = p2;

            // perform sub-block division
            S p = E[p2 - 1];
            D[p2] -= p;
            D[p2 - 1] -= p;
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_SOLVE_KERNEL implements the solver phase of the DC algorithm to
   compute the eigenvalues/eigenvectors of the 'blks' different sub-blocks of a matrix.
        - Call this kernel with batch_count groups in y, and 'blks' groups in x. 
        - Each group will solve a sub-block.  
        - Groups contain a single wavefront **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(WAVEFRONT) 
stedc_solve_kernel(const rocblas_int levs,
                   const rocblas_int blks,
                   const rocblas_int n,
                   S* DD,
                   const rocblas_stride strideD,
                   S* EE,
                   const rocblas_stride strideE,
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
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread index
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tidb_inc = hipBlockDim_x;

    // select batch instance to work with
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    rocblas_int* info = iinfo + bid;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // workspace for solvers
    S* W = WA + bid * (2 * n);

    // Solve the blks sub-blocks in parallel (using classic QR iteration).
    if(sid < blks)
    {
        rocblas_int sbs = ns[sid];  // size of sub-block
        rocblas_int p2 = ps[sid];   // start position of sub-block

        run_steqr(tidb, tidb_inc, sbs, D + p2, E + p2, C + p2 + p2 * ldc, ldc, info, W + p2 * 2,
                  30 * sbs, eps, ssfmin, ssfmax, true);
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGESORT_KERNEL combines the two sorted arrays containing the eigenvalues of 
    every pair of sub-blocks that need to be merged, and gets its corresponding vector z.
        - Call this kernel with batch_count groups in y, and as many groups in x as needed
          to cover the n values of the matrix.
        - Each thread will deal with one value.  
        - Size of groups is set to STEDC_BDIM.**/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
stedc_mergeSort_kernel(const rocblas_int levs,
                       const rocblas_int blks,
                       const rocblas_int k,
                       const rocblas_int n,
                       S* DD,
                       const rocblas_stride strideD,
                       S* CC,
                       const rocblas_int shiftC,
                       const rocblas_int ldc,
                       const rocblas_stride strideC,
                       S* tmpzA,
                       S* vecsA,
                       rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // thread-group id
    rocblas_int gid = hipBlockIdx_x;
    // number of thread-groups
    rocblas_int nofg = hipGridDim_x;
    // thread-group dimension
    rocblas_int dim = hipBlockDim_x;
    // total number of threads
    rocblas_int totdim = nofg * dim;
    // thread id
    rocblas_int tid = gid * dim + hipThreadIdx_x;

    // select batch instance to work with
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // work with all the values (items) in parallel
    for(rocblas_int tx = tid; tx < n; tx += totdim)
    {
        rocblas_int dm = 1 << k;
        rocblas_int dm2 = dm << 1;

        // item 'tx' belongs to sub-block 'bx' and thus participates 
        // in the merge to create the new sub-block 'nbx'
        rocblas_int bx = bisearch(tx, ps, blks, false) - 1;
        rocblas_int nbx = bx / dm2;

        // the new sub-block starts at 'pin', the middle point is 'pmid', and
        // it ends at 'pout'
        rocblas_int tmp = nbx * dm2;
        rocblas_int pin = ps[tmp];
        rocblas_int pmid = ps[tmp + dm];
        tmp += dm2;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;

        // the position where the item 'tx' will end up in the ordered array is 'pos'
        S val = D[tx];
        rocblas_int pos1 = tx < pmid ? bisearch(val, D + pmid, pout - pmid, true) 
                                     : bisearch(val, D + pin, pmid - pin, false);
        rocblas_int pos2 = tx < pmid ? tx - pin : tx - pmid;
        rocblas_int pos = pos1 + pos2;

        // get merged ordered array 'ev' and permutation map 'per'
        rocblas_int* per = pers + pin;
        S* ev = evs + pin;
        ev[pos] = val;
        per[pos] = tx;            

        // get vector Z
        const S inv_sqrt2 = 1 / std::sqrt(2);
        val = tx < pmid ? C[pmid - 1 + tx * ldc] : C[pmid + tx * ldc];
        z[tx] = val * inv_sqrt2;
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEDEFLATE_KERNEL performs deflation for every pair of sub-blocks that need 
    to be merged, and prepare components for secular equations.
        - Call this kernel with batch_count groups in y, and one group in x 
          with at least 'blks' threads.
        - Each thread will deal with one sub-block in the context of the merge**/
template <typename S>
ROCSOLVER_KERNEL void 
stedc_mergeDeflate_kernel(const rocblas_int levs,
                          const rocblas_int blks,
                          const rocblas_int k,
                          const rocblas_int n,
                          S* EE,
                          const rocblas_stride strideE,
                          S* tmpzA,
                          S* vecsA,
                          rocblas_int* splitsA,
                          const S eps)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // thread id
    rocblas_int tx = hipThreadIdx_x;

    // select batch instance to work with
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // temporary arrays in shared memory
    extern __shared__ rocblas_int lsmem[];
    // used to store temp values during the different reductions
    S* shmaxz = reinterpret_cast<S*>(lsmem);
    S* shmaxd = shmaxz + blks;

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // 1. Find tolerance for deflation
    // ---------------------------------------------------
    // 'bx' indexes block 'tx' in the context of the merge according to the level 'k'
    rocblas_int bx = tx % dm2; 
    rocblas_int bs = 0;     // size of sub-block 
    rocblas_int bin = 0;    // initial index
    rocblas_int bout = 0;
    
    // find max values of evs and z in sub-blocks
    S valz, maxz = 0, maxd = 0;
    if(tx < blks)
    {
        bs = ns[tx];
        bin = ps[tx];
        bout = (tx < blks - 1) ? ps[tx + 1] : n;
        for(int ii = 0; ii < bs; ++ii)
        {
            rocblas_int i = ii + bin;
            valz = std::abs(z[i]);
            maxz = (valz > maxz) ? valz : maxz;
        }
        shmaxz[tx] = maxz;
        if(bx == dm2 - 1)
            shmaxd[tx] = abs(evs[bin + bs - 1]);
    }
    __syncthreads();

    // reduction
    rocblas_int dim = dm;
    while(dim > 0)
    {
        if(tx < blks && bx < dim)
        {
            rocblas_int i = tx + dim;
            valz = shmaxz[i];
            maxz = (valz > maxz) ? valz : maxz;
            shmaxz[tx] = maxz;
        }
        dim /= 2;
        __syncthreads();
    }
    maxd = (tx < blks) ? shmaxd[tx + dm2 - 1 - bx] : 0;
    maxz = (tx < blks) ? shmaxz[tx - bx] : 0;
    maxd = (maxz > maxd) ? maxz : maxd;

    // tol should be  8 * eps * (max diagonal or z element participating in the merge)
    S tol = 8 * eps * maxd;


    // 2. Get off-diagonal element 'p' for the merge
    // ----------------------------------------------------------
    // block 'tx' will merge to create the new sub-block 'nbx'
    rocblas_int nbx = tx / dm2;

    // the new sub-block starts at 'pin', the middle point is 'pmid', and
    // it ends at 'pout'. Element 'p' is found at middle point
    rocblas_int pin = 0, pmid = 0, pout = 0;
    S p = 0;
    if(tx < blks)
    {
        rocblas_int tmp = nbx * dm2;
        pin = ps[tmp];
        pmid = ps[tmp + dm];
        tmp += dm2;
        pout = tmp < blks ? ps[tmp] : n;
        p = 2 * E[pmid - 1];
    }



if(tx == 0)
{
    evs[0] = -7;
    evs[1] = -7;
    evs[2] = -7;
    evs[3] = -6;
    evs[4] = -6;
    evs[5] = -6;
    evs[6] = -6;
    evs[7] = -6;
    evs[8] = -5;
    evs[9] = -5;
    evs[10] = -5;
    evs[11] = -4;
    evs[12] = -4;
    evs[13] = -3;
    evs[14] = -2;
    evs[15] = -2;
    evs[16] = 0;
    evs[17] = 0;
    evs[18] = 0;
    evs[19] = 0;
    evs[20] = 1;
}
__syncthreads();

tol = 0;



    // 3. find the position 'pos' where we start looking for deflations in each sub-block
    // ------------------------------------------------------------
    rocblas_int pos = bin;
    if(tx < blks && bx > 0)
    {
        // find the longest sequence that could touch sub-block 'tx'
        bool next = false;
        S tmp = 0;
        S top = evs[pos];
        rocblas_int inc = 1;
        S base = evs[pos - inc];
        while(pos - inc > pin && std::abs(top - base) <= tol)
        {
            inc++;
            base = evs[pos - inc];
        }
        if(inc > 1)
        {
            if(std::abs(top - base) > tol)
            {
                tmp = evs[pos - inc + 1];
                if(std::abs(tmp - base) > tol)
                     next = true;
            }
            else
            {
                tmp = base;
                next = true;
            }
        }
        
        // find first element in sub-block 'tx' outside of the sequence
        if(next)
        {
            pos++;
            top = evs[pos];
            while(pos < bout - 1 && std::abs(top - tmp) <= tol)
            {
                pos++;
                top = evs[pos];
            }
            if(pos >= bout - 1 && std::abs(top - tmp) <= tol)
                pos = -1;
        }
    }
    __syncthreads();


printf("tx = %d --> pos = %d\n",tx,pos);


    // 4. Deflate values and compute corresponding rotations.
    //    (work with each sub-block in parallel when possible)
    // ----------------------------------------------------------------




}



//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPARE_KERNEL performs deflation and prepares the secular equation for
    every pair of sub-blocks that need to be merged. 
        - Call this kernel with batch_count groups in y, and as many groups as half of the 
          unmerged sub-blocks in current level in x. Each group works with a merge of a pair
          of sub-blocks. Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergePrepare_kernel(const rocblas_int levs,
                              const rocblas_int blks,
                              const rocblas_int k,
                              const rocblas_int n,
                              S* DD,
                              const rocblas_stride strideD,
                              S* EE,
                              const rocblas_stride strideE,
                              S* CC,
                              const rocblas_int shiftC,
                              const rocblas_int ldc,
                              const rocblas_stride strideC,
                              S* tmpzA,
                              S* vecsA,
                              rocblas_int* splitsA,
                              const S eps)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tid, tx;

    // select batch instance to work with
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // temporary arrays in shared memory
    // used to store temp values during the different reductions
    extern __shared__ rocblas_int lsmem[];
    S* inrmsd = reinterpret_cast<S*>(lsmem);
    S* inrmsz = inrmsd + hipBlockDim_x;

    // tn is the number of thread-groups needed in level k of the merge
    rocblas_int bd = 1 << k;
    rocblas_int bdm = bd << 1;
    rocblas_int tn = blks / bdm;

    // Work with merges on level k. A thread-group works with two leaves in the merge tree.
    if(sid < tn)
    {
        rocblas_int iam, sz, dim, dim2, p2;

        // tid indexes the sub-blocks in the entire matrix
        // iam indexes the sub-blocks in the context of the merge
        // (according to its level in the merge tree)
        dim = hipBlockDim_x / 2;
        iam = tidb / dim;
        tx = tidb % dim;
        tid = sid * bdm + iam * bd;
        p2 = ps[tid];

        // 1. find rank-1 modification components (z and p) for this merge
        // ----------------------------------------------------------------
        // Threads with iam = 0 work with components below the merge point;
        // threads with iam = 1 work above the merge point
        sz = ns[tid];
        for(int j = 1; j < bd; ++j)
            sz += ns[tid + j];
        // with this, all threads involved in a merge
        // will point to the same row of C and the same off-diag element
        S* ptz = (iam == 0) ? C + p2 - 1 + sz : C + p2;
        S p = (iam == 0) ? 2 * E[p2 - 1 + sz] : 2 * E[p2 - 1];

        // copy elements of z
        for(int j = tx; j < sz; j += dim)
            z[p2 + j] = ptz[(p2 + j) * ldc] / sqrt(2);


        // 2. calculate deflation tolerance
        // ----------------------------------------------------------------
        // compute maximum of diagonal and z in each merge block
        S valf, valg, maxd, maxz;
        maxd = 0;
        maxz = 0;
        for(int i = tx; i < sz; i += dim)
        {
            valf = std::abs(D[p2 + i]);
            valg = std::abs(z[p2 + i]);
            maxd = valf > maxd ? valf : maxd;
            maxz = valg > maxz ? valg : maxz;
        }
        inrmsd[tidb] = maxd;
        inrmsz[tidb] = maxz;
        __syncthreads();

        dim2 = dim;
        while(dim2 > 0)
        {
            if(tidb < dim2)
            {
                valf = inrmsd[tidb + dim2];
                valg = inrmsz[tidb + dim2];
                maxd = valf > maxd ? valf : maxd;
                maxz = valg > maxz ? valg : maxz;
                inrmsd[tidb] = maxd;
                inrmsz[tidb] = maxz;
            }
            dim2 /= 2;
            __syncthreads();
        }

        // tol should be  8 * eps * (max diagonal or z element participating in merge)
        maxd = inrmsd[0];
        maxz = inrmsz[0];
        maxd = maxz > maxd ? maxz : maxd;
        S tol = 8 * eps * maxd;


        // 3. deflate eigenvalues
        // ----------------------------------------------------------------
        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position.
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        rocblas_int in = tid - iam * bd;
        sz = ns[in];
        for(int i = 1; i < bdm; ++i)
            sz += ns[in + i];
        in = ps[in];

        // first deflate zero components
        S f, g, c, s, rr;
        for(int i = tidb; i < sz; i += hipBlockDim_x)
        {
            tx = in + i;
            g = z[tx];
            if(abs(p * g) <= tol)
                // deflated ev because component in z is zero
                idd[tx] = 0;
            else
                idd[tx] = 1;
        }
        __syncthreads();

        // now deflate repeated values
        rocblas_int sz_even, sz_half, base, top, com;
        sz_even = (sz % 2 == 1) ? sz + 1 : sz;
        sz_half = sz_even / 2;

        // the number of rounds needed is sz_even - 1
        for(int r = 0; r < sz_even - 1; ++r)
        {
            // in each round threads analyze pairs of values in parallel
            // sz_half pairs are needed
            for(int i = tidb; i < sz_half; i += hipBlockDim_x)
            {
                // determine pair of values (base, top)
                com = 2 * (i - r);
                base = (i == 0)             ? 0
                    : (r < i)               ? com
                    : (r > i - 1 + sz_half) ? 2 * (sz_even - 1) + com
                                            : 1 - com;

                com = 2 * (i + r);
                top = (r < sz_half - i)     ? 1 + com
                    : (r > sz_even - 2 - i) ? 3 - 2 * sz_even + com
                                            : 2 * (sz_even - 1) - com;

                if(base > top)
                {
                    com = base;
                    base = top;
                    top = com;
                }

                // compare values and deflate if needed
                base += in;
                top += in;
                if(idd[base] == 1 && idd[top] == 1 && top < sz + in)
                {
                    if(abs(D[base] - D[top]) <= tol)
                    {
                        // deflated ev because it is repeated
                        idd[top] = 0;
                        // rotation to eliminate component in z
                        g = z[top];
                        f = z[base];
                        lartg(f, g, c, s, rr);
                        z[base] = rr;
                        z[top] = 0;
                        // update C with the rotation
                        for(int ii = 0; ii < n; ++ii)
                        {
                            valf = C[ii + base * ldc];
                            valg = C[ii + top * ldc];
                            C[ii + base * ldc] = valf * c - valg * s;
                            C[ii + top * ldc] = valf * s + valg * c;
                        }
                    }
                }
                __syncthreads();
            }
        }


        // 4. Organize data with non-deflated values to prepare secular equation
        // ------------------------------------------------------------------------ 
        // define shifted arrays
        S* tmpd = temps + in * n;
        S* diag = D + in;
        rocblas_int* mask = idd + in;
        S* zz = z + in;
        rocblas_int* per = pers + in;
        S* ev = evs + in;

        // find degree and components of secular equation
        // tmpd contains the non-deflated diagonal elements (ie. poles of the
        // secular eqn) zz contains the corresponding non-zero elements of the
        // rank-1 modif vector
        rocblas_int dd = 0;
        for(int i = 0; i < sz; ++i)
        {
            if(mask[i] == 1)
            {
                if(tidb == 0)
                {
                    per[dd] = i;
                    tmpd[dd] = p < 0 ? -diag[i] : diag[i];
                    if(dd != i)
                        zz[dd] = zz[i];
                }
                dd++;
            }
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVALUES_KERNEL solves the secular equation for every pair of sub-blocks 
    that need to be merged. 
        - Call this kernel with batch_count groups in y, and as many groups as half of the 
          unmerged sub-blocks in current level in x. Each group works with a merge of a pair
          of sub-blocks. Groups are size STEDC_BDIM **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeValues_kernel(const rocblas_int levs,
                             const rocblas_int blks,
                             const rocblas_int k,
                             const rocblas_int n,
                             S* DD,
                             const rocblas_stride strideD,
                             S* EE,
                             const rocblas_stride strideE,
                             S* tmpzA,
                             S* vecsA,
                             rocblas_int* splitsA,
                             const S eps,
                             const S ssfmin,
                             const S ssfmax)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tid;

    // select batch instance to work with
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // tn is the number of thread-groups needed in level k of the merge
    rocblas_int bd = 1 << k;
    rocblas_int bdm = bd << 1;
    rocblas_int tn = blks / bdm;

    // Work with merges on level k. A thread-group works with two leaves in the merge tree;
    // all threads work together to solve the secular equation.
    if(sid < tn)
    {
        rocblas_int iam, sz, dim, p2;
        S valf, valg;

        // tid indexes the sub-blocks in the entire split block
        // iam indexes the sub-blocks in the context of the merge
        // (according to its level in the merge tree)
        dim = hipBlockDim_x / 2;
        iam = tidb / dim;
        tid = sid * bdm + iam * bd;
        p2 = ps[tid];

        // Find off-diagonal element of the merge
        // Threads with iam = 0 work with components below the merge point;
        // threads with iam = 1 work above the merge point
        sz = ns[tid];
        for(int j = 1; j < bd; ++j)
            sz += ns[tid + j];
        // with this, all threads involved in a merge
        // will point to the same row of C and the same off-diag element
        S p = (iam == 0) ? 2 * E[p2 - 1 + sz] : 2 * E[p2 - 1];

        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position.
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        rocblas_int in = tid - iam * bd;
        sz = ns[in];
        for(int i = 1; i < bdm; ++i)
            sz += ns[in + i];
        in = ps[in];


        // 1. Organize data with non-deflated values to prepare secular equation
        // ----------------------------------------------------------------- 
        // All threads of the group participating in the merge will work together
        // to solve the correspondinbg secular eqn. Now 'iam' indexes those threads
        iam = tidb;
        bdm = hipBlockDim_x;

        // define shifted arrays
        S* tmpd = temps + in * n;
        S* ev = evs + in;
        S* diag = D + in;
        rocblas_int* mask = idd + in;
        S* zz = z + in;
        rocblas_int* per = pers + in;

        // find degree of secular equation
        rocblas_int dd = 0;
        for(int i = 0; i < sz; ++i)
        {
            if(mask[i] == 1)
                dd++;
        }

        // Order the elements in tmpd and zz using a simple parallel selection/bubble sort.
        // This will allow us to find initial intervals for eigenvalue guesses
        for(int i = 0; i < dd; i++)
        {
            for(int j = 2 * iam + i % 2; j < dd - 1; j += 2 * bdm)
            {
                if(tmpd[j] > tmpd[j + 1])
                {
                    swap(tmpd[j], tmpd[j + 1]);
                    swap(zz[j], zz[j + 1]);
                    swap(per[j], per[j + 1]);
                }
            }
            __syncthreads();
        }

        // make dd copies of the non-deflated ordered diagonal elements
        // (i.e. the poles of the secular eqn) so that the distances to the
        // eigenvalues (D - lambda_i) are updated while computing each eigenvalue.
        // This will prevent collapses and division by zero when an eigenvalue
        // is too close to a pole.
        for(int i = iam; i < dd; i += bdm)
        {
            for(int j = i + n; j < i + sz * n; j += n)
                tmpd[j] = tmpd[i];
        }

        // finally copy over all diagonal elements in ev. ev will be overwritten
        // by the new computed eigenvalues of the merged block
        for(int i = iam; i < sz; i += bdm)
            ev[i] = diag[i];
        __syncthreads();


        // 2. Solve secular eqns, i.e. find the dd zeros
        // corresponding to non-deflated new eigenvalues of the merged block
        // ----------------------------------------------------------------- 
        // each thread will find a different zero in parallel
        S a, b;
        for(int j = iam; j < sz; j += bdm)
        {
            if(mask[j] == 1)
            {
                // find position in the ordered array
                valf = p < 0 ? -ev[j] : ev[j];
                int count = dd, cc = 0;
                while(count > 0)
                {
                    auto step = count / 2;
                    auto it = cc + step;
                    if(tmpd[it + j * n] < valf)
                    {
                        cc = ++it;
                        count -= step + 1;
                    }
                    else
                        count = step;
                }

                // computed zero will overwrite 'ev' at the corresponding position.
                // 'tmpd' will be updated with the distances D - lambda_i.
                // deflated values are not changed.
                rocblas_int linfo;

#if defined(ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER)
                linfo = slaed4(dd, cc, tmpd + j * n, zz, std::abs(p), ev[j]);
#else
                if(cc == dd - 1)
                    linfo = seq_solve_ext(dd, tmpd + j * n, zz, (p < 0 ? -p : p), ev + j, eps,
                                          ssfmin, ssfmax);
                else
                    linfo = seq_solve(dd, tmpd + j * n, zz, (p < 0 ? -p : p), cc, ev + j, eps,
                                      ssfmin, ssfmax);
#endif
                if(p < 0)
                    ev[j] *= -1;
            }
        }
        __syncthreads();

        // Re-scale vector Z to avoid bad numerics when an eigenvalue
        // is too close to a pole
        for(int i = iam; i < dd; i += bdm)
        {
            valf = 1;
            for(int j = 0; j < sz; ++j)
            {
                if(mask[j] == 1)
                {
                    valg = tmpd[i + j * n];
                    valf *= (per[i] == j) ? valg : valg / (diag[per[i]] - diag[j]);
                }
            }
            valf = sqrt(std::abs(valf));
            zz[i] = zz[i] < 0 ? -valf : valf;
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVECTORS_KERNEL prepares vectors from the secular equation for
    every pair of sub-blocks that need to be merged.
        - Call this kernel with batch_count groups in y, and as many groups as columns would 
          be in the matrix if its size is exact multiple of the number of sub-blocks 'blks'.
          Each group works with a column. Groups are size STEDC_BDIM.
        - If a group has an id larger than the actual number of columns it will do nothing. **/
template <bool USEGEMM, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeVectors_kernel(const rocblas_int levs,
                              const rocblas_int blks,
                              const rocblas_int k,
                              const rocblas_int n,
                              S* DD,
                              const rocblas_stride strideD,
                              S* EE,
                              const rocblas_stride strideE,
                              S* CC,
                              const rocblas_int shiftC,
                              const rocblas_int ldc,
                              const rocblas_stride strideC,
                              S* tmpzA,
                              S* vecsA,
                              rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid, vidb;

    // select batch instance to work with
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // temporary arrays in shared memory
    // used to store temp values during the different reductions
    extern __shared__ rocblas_int lsmem[];
    S* inrms = reinterpret_cast<S*>(lsmem);

    // tn is max number of vectors in each sub-block
    rocblas_int bd = 1 << k;
    rocblas_int bdm = bd << 1;
    rocblas_int tn = (n - 1) / blks + 1;

    // Work with merges on level k. Each thread-group works with one vector.
    if(sid < tn * blks)
    {
        rocblas_int iam, sz, p2;
        S valf, valg;

        // tid indexes the sub-blocks in the entire split block
        tid = sid / tn;
        p2 = ps[tid];
        // vidb indexes the vectors associated with each sub-block
        vidb = sid % tn;
        // iam indexes the sub-blocks in the context of the merge
        // (according to its level in the merge tree)
        iam = tid % bdm;

        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position
        rocblas_int in = ps[tid - iam];
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        sz = ns[tid];
        for(int i = iam; i > 0; --i)
            sz += ns[tid - i];
        for(int i = bdm - 1 - iam; i > 0; --i)
            sz += ns[tid + i];

        // define shifted arrays
        S* tmpd = temps + in * n;
        S* ev = evs + in;
        S* diag = D + in;
        rocblas_int* mask = idd + in;
        S* zz = z + in;
        rocblas_int* per = pers + in;

        // find degree of secular equation
        rocblas_int dd = 0;
        for(int i = 0; i < sz; ++i)
        {
            if(mask[i] == 1)
                dd++;
        }
        __syncthreads();


        // Prepare vectors corresponding to non-deflated values
        S temp, nrm;
        rocblas_int j = vidb;
        bool go = (j < ns[tid]);
        S* putvec = USEGEMM ? vecs : temps;

        if(go)
        {
            if(idd[p2 + j] == 1)
            {
                // compute vectors of rank-1 perturbed system and their norms
                nrm = 0;
                for(int i = tidb; i < dd; i += dim)
                {
                    valf = zz[i] / temps[i + (p2 + j) * n];
                    nrm += valf * valf;
                    putvec[i + (p2 + j) * n] = valf;
                }
                inrms[tidb] = nrm;
                __syncthreads();

                // reduction (for the norms)
                for(int r = dim / 2; r > 0; r /= 2)
                {
                    if(tidb < r)
                    {
                        nrm += inrms[tidb + r];
                        inrms[tidb] = nrm;
                    }
                    __syncthreads();
                }
                nrm = sqrt(inrms[0]);
            }

            if(USEGEMM)
            {
                // when using external gemms for the update, we need to
                // put vectors in padded matrix 'temps'
                // (this is to compute 'vecs = C * temps' using external gemm call)
                for(int i = tidb; i < in + sz; i += dim)
                {
                    if(i >= in && idd[p2 + j] == 1 && idd[i] == 1)
                    {
                        dd = 0;
                        for(int k = in; k < i; ++k)
                        {
                            if(idd[k] == 0)
                                dd++;
                        }
                        temps[pers[i - dd] + in + (p2 + j) * n]
                            = vecs[i - dd - in + (p2 + j) * n] / nrm;
                    }
                    else
                        temps[i + (p2 + j) * n] = 0;
                }
            }
            else
            {
                // otherwise, use internal gemm-like procedure to
                // multiply by C (row by row)
                rocblas_int tsz = 1 << (levs - 1 - k);
                tsz = (n - 1) / tsz + 1;
                if(idd[p2 + j] == 1)
                {
                    for(int ii = 0; ii < tsz; ++ii)
                    {
                        rocblas_int i = in + ii;

                        // inner products
                        temp = 0;
                        if(ii < sz)
                        {
                            for(int kk = tidb; kk < dd; kk += dim)
                                temp += C[i + (per[kk] + in) * ldc] * temps[kk + (p2 + j) * n];
                        }
                        inrms[tidb] = temp;
                        __syncthreads();

                        // reduction
                        for(int r = dim / 2; r > 0; r /= 2)
                        {
                            if(ii < sz && tidb < r)
                            {
                                temp += inrms[tidb + r];
                                inrms[tidb] = temp;
                            }
                            __syncthreads();
                        }

                        // result
                        if(ii < sz && tidb == 0)
                            vecs[i + (p2 + j) * n] = temp / nrm;
                        __syncthreads();
                    }
                }
            }
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEUPDATE_KERNEL updates vectors and values after a merge is done. 
        - Call this kernel with batch_count groups in y, and as many groups as columns would 
          be in the matrix if its size is exact multiple of the number of sub-blocks 'blks'.
          Each group works with a column. Groups are size STEDC_BDIM.
        - If a group has an id larger than the actual number of columns it will do nothing. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeUpdate_kernel(const rocblas_int levs,
                             const rocblas_int blks,
                             const rocblas_int k,
                             const rocblas_int n,
                             S* DD,
                             const rocblas_stride strideD,
                             S* CC,
                             const rocblas_int shiftC,
                             const rocblas_int ldc,
                             const rocblas_stride strideC,
                             S* tmpzA,
                             S* vecsA,
                             rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int sid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid, vidb;

    // select batch instance to work with
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);

    // tn is max number of vectors in each sub-block
    rocblas_int bd = 1 << k;
    rocblas_int bdm = bd << 1;
    rocblas_int tn = (n - 1) / blks + 1;

    // Work with merges on level k. Each thread-group works with one vector.
    if(sid < tn * blks)
    {
        rocblas_int iam, sz, p2;
        S valf, valg;

        // tid indexes the sub-blocks in the entire split block
        tid = sid / tn;
        p2 = ps[tid];
        // vidb indexes the vectors associated with each sub-block
        vidb = sid % tn;
        // iam indexes the sub-blocks in the context of the merge
        // (according to its level in the merge tree)
        iam = tid % bdm;

        // determine boundaries of what would be the new merged sub-block
        // 'in' will be its initial position
        rocblas_int in = ps[tid - iam];
        // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
        sz = ns[tid];
        for(int i = iam; i > 0; --i)
            sz += ns[tid - i];
        for(int i = bdm - 1 - iam; i > 0; --i)
            sz += ns[tid + i];

        // update D and C with computed values and vectors
        rocblas_int j = vidb;
        bool go = (j < ns[tid] && idd[p2 + j] == 1);
        if(go)
        {
            if(tidb == 0)
                D[p2 + j] = evs[p2 + j];
            for(int i = in + tidb; i < in + sz; i += dim)
                C[i + (p2 + j) * ldc] = vecs[i + (p2 + j) * n];
        }
    }
}


/** STEDC_SORT sorts computed eigenvalues and eigenvectors in increasing order **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS1) stedc_sort(const rocblas_int n,
                                                        S* DD,
                                                        const rocblas_stride strideD,
                                                        U CC,
                                                        const rocblas_int shiftC,
                                                        const rocblas_int ldc,
                                                        const rocblas_stride strideC,
                                                        const rocblas_int batch_count,
                                                        rocblas_int* work,
                                                        rocblas_int* nev = nullptr)
{
    // -----------------------------------
    // use z-grid dimension as batch index
    // -----------------------------------
    rocblas_int bid_start = hipBlockIdx_z;
    rocblas_int bid_inc = hipGridDim_z;

    int tid = hipThreadIdx_x;

    rocblas_int* const map = work + bid_start * ((int64_t)n);

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        // ---------------------------------------------
        // select batch instance to work with
        // (avoiding arithmetics with possible nullptrs)
        // ---------------------------------------------
        T* C = nullptr;
        if(CC)
            C = load_ptr_batch<T>(CC, bid, shiftC, strideC);
        S* D = DD + (bid * strideD);
        rocblas_int nn;
        if(nev)
            nn = nev[bid];
        else
            nn = n;

        bool constexpr use_shell_sort = true;

        __syncthreads();

        if(use_shell_sort)
            shell_sort(nn, D, map);
        else
            selection_sort(nn, D, map);
        __syncthreads();

        permute_swap(n, C, ldc, map, nn);
        __syncthreads();
    }
}


/******************* Host functions *********************************************/
/*******************************************************************************/

//--------------------------------------------------------------------------------------//
/** STEDC_NUM_LEVELS returns the ideal number of times/levels in which a matrix
    will be divided during the divide phase of divide & conquer algorithm 
    i.e. number of sub-blocks = 2^levels **/
inline rocblas_int stedc_num_levels(const rocblas_int n)
{
    rocblas_int levels;

    if(n <= 16)
        levels = 0;
    else
        levels = std::ceil(std::log2(n)) - 4;

//   return levels;
    return 3;
}

//--------------------------------------------------------------------------------------//
/** This helper calculates required workspace size **/
template <bool BATCHED, typename T, typename S>
void rocsolver_stedc_getMemorySize(const rocblas_evect evect,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work_stack,
                                   size_t* size_tempvect,
                                   size_t* size_tempgemm,
                                   size_t* size_tmpz,
                                   size_t* size_splits_map,
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
        *size_splits_map = 0;
        *size_tmpz = 0;
        return;
    }

    // if no eigenvectors required with classic solver
    if(evect == rocblas_evect_none)
    {
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits_map = 0;
        *size_tmpz = 0;
        rocsolver_sterf_getMemorySize<S>(n, batch_count, size_work_stack);
    }

    // if size is too small with classic solver
    else if(n < STEDC_MIN_DC_SIZE)
    {
        *size_tempvect = 0;
        *size_tempgemm = 0;
        *size_workArr = 0;
        *size_splits_map = 0;
        *size_tmpz = 0;
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);
    }

    // otherwise use divide and conquer algorithm:
    else
    {
        // requirements for solver of small independent blocks
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);

        // extra requirements for original eigenvectors of small independent blocks
        if(evect != rocblas_evect_tridiagonal)
            *size_tempvect = sizeof(S) * (n * n) * batch_count;
        else
            *size_tempvect = 0;
        *size_tempgemm = sizeof(S) * 2 * (n * n) * batch_count;
        if(BATCHED && !COMPLEX)
            *size_workArr = sizeof(S*) * batch_count;
        else
            *size_workArr = 0;

        // size for split blocks and sub-blocks positions
        *size_splits_map = sizeof(rocblas_int) * (5 * n + 2) * batch_count;

        // size for temporary diagonal and rank-1 modif vector
        *size_tmpz = sizeof(S) * (2 * n) * batch_count;
    }
}

//--------------------------------------------------------------------------------------//
/** This helper check argument correctness for stedc API **/
template <typename T, typename S>
rocblas_status rocsolver_stedc_argCheck(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_int n,
                                        S D,
                                        S E,
                                        T C,
                                        const rocblas_int ldc,
                                        rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(evect != rocblas_evect_none && evect != rocblas_evect_tridiagonal
       && evect != rocblas_evect_original)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0)
        return rocblas_status_invalid_size;
    if(evect != rocblas_evect_none && ldc < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n > 1 && !E) || (evect != rocblas_evect_none && n && !C) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//--------------------------------------------------------------------------------------//
/** STEDC templated function **/
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_stedc_template(rocblas_handle handle,
                                        const rocblas_evect evect,
                                        const rocblas_int n,
                                        S* D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        U C,
                                        const rocblas_int shiftC,
                                        const rocblas_int ldc,
                                        const rocblas_stride strideC,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        void* work_stack,
                                        S* tempvect,
                                        S* tempgemm,
                                        S* tmpz,
                                        rocblas_int* splits,
                                        S** workArr)
{
    ROCSOLVER_ENTER("stedc", "evect:", evect, "n:", n, "shiftD:", shiftD, "shiftE:", shiftE,
                    "shiftC:", shiftC, "ldc:", ldc, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    auto const splits_map = splits;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threads(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 1 && evect != rocblas_evect_none)
        ROCSOLVER_LAUNCH_KERNEL(reset_batch_info<T>, dim3(1, batch_count), dim3(1, 1), 0, stream, C,
                                strideC, n, 1);
    if(n <= 1)
        return rocblas_status_success;

    // if no eigenvectors required with the classic solver, use sterf
    if(evect == rocblas_evect_none)
    {
        rocsolver_sterf_template<S>(handle, n, D, shiftD, strideD, E, shiftE, strideE, info,
                                    batch_count, static_cast<rocblas_int*>(work_stack));
    }

    // if size is too small with classic solver, use steqr
    else if(n < STEDC_MIN_DC_SIZE)
    {
        rocsolver_steqr_template<T>(handle, evect, n, D, shiftD, strideD, E, shiftE, strideE, C,
                                    shiftC, ldc, strideC, info, batch_count, work_stack);
    }

    // otherwise use divide and conquer algorithm:
    else
    {

print_device_matrix(std::cout,"D in",1,n,D,1);
print_device_matrix(std::cout,"E in",1,n-1,E,1);

        // initialize temporary array for vector updates
        size_t size_tempgemm = sizeof(S) * 2 * n * n * batch_count;
        HIP_CHECK(hipMemsetAsync((void*)tempgemm, 0, size_tempgemm, stream));

        // everything must be executed with scalars on the host
        rocblas_pointer_mode old_mode;
        rocblas_get_pointer_mode(handle, &old_mode);
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
        S one = 1.0;
        S zero = 0.0;

        // constants
        S eps = get_epsilon<S>();
        S ssfmin = get_safemin<S>();
        S ssfmax = S(1.0) / ssfmin;
        ssfmin = sqrt(ssfmin) / (eps * eps);
        ssfmax = sqrt(ssfmax) / S(3.0);

        // find number of sub-blocks 
        rocblas_int levs = stedc_num_levels(n);
        rocblas_int blks = 1 << levs;

        // initialize identity matrix in V
        // if evect is tridiagonal we can store V directly in C
        // otherwise, they must be kept separate to compute C*V
        S* V = tempvect;
        rocblas_int ldv = n;
        rocblas_stride strideV = n * n;
        if(evect == rocblas_evect_tridiagonal)
        {
            V = (S*)(C + shiftC);
            ldv = (rocblas_int)(sizeof(T) / sizeof(S)) * ldc;
            strideV = (rocblas_int)(sizeof(T) / sizeof(S)) * strideC;
        }
        rocblas_int groupsn = (n - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL(init_ident<S>, dim3(groupsn, groupsn, batch_count), dim3(BS2, BS2),
                                0, stream, n, n, V, 0, ldv, strideV);

        // 1. divide phase
        //-----------------------------
        rocblas_int groups = (batch_count - 1) / STEDC_BDIM + 1;
        ROCSOLVER_LAUNCH_KERNEL((stedc_divide_kernel<S>),
                                dim3(groups), dim3(STEDC_BDIM), 0, stream, levs, blks, n, D + shiftD,
                                strideD, E + shiftE, strideE, batch_count, splits);

        // 2. solve phase
        //-----------------------------
        ROCSOLVER_LAUNCH_KERNEL((stedc_solve_kernel<S>),
                                dim3(blks, batch_count), dim3(WAVEFRONT), 0, stream, levs, blks, 
                                n, D + shiftD, strideD, E + shiftE, strideE, 
                                V, 0, ldv, strideV, info, (S*)work_stack, splits, 
                                eps, ssfmin, ssfmax);

print_device_matrix(std::cout,"D at leaves",1,n,D,1);
print_device_matrix(std::cout,"E at leaves",1,n-1,E,1);
print_device_matrix(std::cout,"V at leaves",n,n,V,ldv);


        // 3. merge phase
        //----------------
        size_t lmemsize = sizeof(S) * blks;
        size_t lmemsize1 = sizeof(S) * 2 * STEDC_BDIM;
        size_t lmemsize3 = sizeof(S) * STEDC_BDIM;
        rocblas_int numgrps3 = ((n - 1) / blks + 1) * blks;

        // launch merge for level k
        for(rocblas_int k = 0; k < levs; ++k)
        {
            // a. prepare secular equations
            rocblas_int numgrps2 = (n - 1) / STEDC_BDIM + 1;

printf("start merge at level k = %d\n",k);
printf("------------------------------------------\n\n");
print_device_matrix(std::cout,"splits",1,n+2,splits,1);
print_device_matrix(std::cout,"ns",1,n,splits+n+2,1);
print_device_matrix(std::cout,"ps",1,n,splits+2*n+2,1);
print_device_matrix(std::cout,"D to be sorted",1,n,D,1);

            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeSort_kernel<S>), dim3(numgrps2, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, D + shiftD, strideD,
                                    V, 0, ldv, strideV, tmpz, tempgemm, splits);

print_device_matrix(std::cout,"Z",1,n,tmpz,1);
print_device_matrix(std::cout,"sorted D",1,n,tmpz+n,1);
print_device_matrix(std::cout,"pers",1,n,splits+4*n+2,1);
            
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeDeflate_kernel<S>), dim3(1, batch_count),
                                    dim3(64), lmemsize, stream, levs, blks, k, n, E + shiftE, strideE,
                                    tmpz, tempgemm, splits, eps);


            numgrps2 = 1 << (levs - 1 - k);
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_kernel<S>),
                                    dim3(numgrps2, batch_count), dim3(STEDC_BDIM), lmemsize1, stream, 
                                    levs, blks, k, n, D + shiftD, strideD,
                                    E + shiftE, strideE, V, 0, ldv, strideV, tmpz, tempgemm, splits,
                                    eps);

            // b. solve secular eq to find merged eigenvalues
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_kernel<S>),
                                    dim3(numgrps2, batch_count), dim3(STEDC_BDIM), 0, stream, 
                                    levs, blks, k, n, D + shiftD, strideD,
                                    E + shiftE, strideE, tmpz, tempgemm, splits, eps, ssfmin, ssfmax);

            // c. find merged eigenvectors
            ROCSOLVER_LAUNCH_KERNEL(
                (stedc_mergeVectors_kernel<STEDC_EXTERNAL_GEMM, S>),
                dim3(numgrps3, batch_count), dim3(STEDC_BDIM), lmemsize3, stream, 
                levs, blks, k, n, D + shiftD, strideD, E + shiftE, strideE, V, 0, ldv, strideV, 
                tmpz, tempgemm, splits);

            if(STEDC_EXTERNAL_GEMM)
            {
                // using external gemms with padded matrices to do the vector update
                // One single full gemm of size n x n x n merges all the blocks in the level
                // TODO: using macro STEDC_EXTERNAL_GEMM = true for now. In the future we can pass
                // STEDC_EXTERNAL_GEMM at run time to switch between internal vector updates and
                // external gemm based updates.
                rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n,
                               &one, V, 0, ldv, strideV, tempgemm, n * n, n, 2 * n * n, &zero,
                               tempgemm, 0, n, 2 * n * n, batch_count, workArr);
            }

            // d. update level
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeUpdate_kernel<S>),
                                    dim3(numgrps3, batch_count), dim3(STEDC_BDIM), 0, stream, 
                                    levs, blks, k, n, D + shiftD, strideD,
                                    V, 0, ldv, strideV, tmpz, tempgemm, splits);
        }

        // 4. update and sort
        //----------------------
        if(evect != rocblas_evect_tridiagonal)
        {
            // eigenvectors C <- C*V
            local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, V, tempgemm,
                                            tempgemm + strideV, 0, ldv, strideV, batch_count,
                                            workArr);
        }
        else if constexpr(rocblas_is_complex<T>)
        {
            // V is stored in C but is of type S; need to convert to type T
            // tempgemm = V
            ROCSOLVER_LAUNCH_KERNEL(copy_mat<S>, dim3(groupsn, groupsn, batch_count), dim3(BS2, BS2),
                                    0, stream, copymat_to_buffer, n, n, V, 0, ldv, strideV, tempgemm);

            // imag(C) = zeros
            ROCSOLVER_LAUNCH_KERNEL(set_zero<T>, dim3(groupsn, groupsn, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, C, shiftC, ldc, strideC);

            // real(C) = tempgemm
            ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(groupsn, groupsn, batch_count),
                                    dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, C, shiftC,
                                    ldc, strideC, tempgemm);
        }

        // finally sort eigenvalues and eigenvectors
        ROCSOLVER_LAUNCH_KERNEL((stedc_sort<T>), dim3(1, 1, batch_count), dim3(BS1), 0, stream, n,
                                D + shiftD, strideD, C, shiftC, ldc, strideC, batch_count,
                                splits_map);

        rocblas_set_pointer_mode(handle, old_mode);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
