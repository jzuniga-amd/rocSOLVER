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
        rocblas_int* splits = splitsA + bid * (5 * n + blks);
        // the sub-blocks sizes
        rocblas_int* ns = splits + n;
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
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
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
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);

    // work with all the values (items) in parallel
    for(rocblas_int tx = tid; tx < n; tx += totdim)
    {
        rocblas_int dm = 1 << k;
        rocblas_int dm2 = dm << 1;

        // item 'tx' belongs to sub-block 'bx' and thus participates 
        // in the merge to create the new sub-block 'nbx'
        rocblas_int bx = bisearch(tx, ps, blks, false, false) - 1;
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
        rocblas_int pos1 = tx < pmid ? bisearch(val, D + pmid, pout - pmid, true, false) 
                                     : bisearch(val, D + pin, pmid - pin, false, false);
        rocblas_int pos2 = tx < pmid ? tx - pin : tx - pmid;
        rocblas_int pos = pos1 + pos2;

        // get merged ordered array 'ev' and permutation map 'per'
        rocblas_int* per = pers + pin;
        S* ev = vecs + n + pin;
        ev[pos] = val;
        per[pos] = tx;            

        // get vector Z
        const S inv_sqrt2 = 1 / std::sqrt(2);
        S* z = vecs;
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
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    rocblas_int* nrs = pers + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (3 * n);
    // roots of secular equations
    S* evr = z + n;
    S* evf = z + 2*n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // temporary arrays in shared memory
    extern __shared__ rocblas_int lsmem[];
    // used to store temp values during the different reductions
    S* shmaxz = reinterpret_cast<S*>(lsmem);
    S* shmaxd = shmaxz + blks;

    
    // 1. Get off-diagonal element 'p' for the merge
    // ----------------------------------------------------------
    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;
    // size of sub-block 'tx'
    rocblas_int bs = ns[tx];
    // its initial index      
    rocblas_int bin = ps[tx];
    // 'bx' indexes block 'tx' in the context of the merge according to the level 'k'
    rocblas_int bx = tx % dm2; 
    // block 'tx' will be merge to create the new sub-block 'nbx'
    rocblas_int nbx = tx / dm2;

    // the new sub-block of size 'sz' starts at 'pin', the middle point is 'pmid', and
    // it ends at 'pout'. Element 'p' is found at middle point
    rocblas_int pin = 0, pmid = 0, pout = 0, sz = 0;
    S p = 0;
    if(tx < blks)
    {
        rocblas_int tmp = nbx * dm2;
        pin = ps[tmp];
        pmid = ps[tmp + dm];
        tmp += dm2;
        pout = tmp < blks ? ps[tmp] : n;
        p = 2 * E[pmid - 1];
        sz = pout - pin;
    }


    // 2. Find tolerance for deflation
    // ---------------------------------------------------
    // find max values of evs and z in the sub-blocks
    S* zz = vecs;
    S* vals = vecs + n;
    S* ztmp = vecs + 2 * n;
    S valz, vald, maxz = 0, maxd = 0;
    if(tx < blks)
    {
        for(int ii = 0; ii < bs; ++ii)
        {
            rocblas_int i = ii + bin;
            valz = std::abs(zz[i]);
            maxz = (valz > maxz) ? valz : maxz;
        }
        shmaxz[tx] = maxz;
        if(bx == 0)
        {
            maxd = abs(vals[pin]);
            vald = abs(vals[pout - 1]);
            shmaxd[tx] = (vald > maxd) ? vald : maxd;
        }
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
    maxd = (tx < blks) ? shmaxd[tx - bx] : 0;
    maxz = (tx < blks) ? shmaxz[tx - bx] : 0;
    maxd = (maxz > maxd) ? maxz : maxd;

    // tol should be  8 * eps * (max diagonal or z element participating in the merge)
    S tol = 8 * eps * maxd;


    // 3. Deflate values and compute corresponding rotations.
    // ----------------------------------------------------------------
    rocblas_int nf = 0;         // number of deflated values
    rocblas_int nr = 0;         // number of non-deflated (remaining values)
    
    if(tx < blks && bx == 0)
    {    
        // arrays for the sub-block:
        // 'evrf' has the form [remaining values | deflated values].
        // 'idrf' contains the corresponding indices, and
        // 'z' the non-zeroed elements of z.
        rocblas_int* idrf = idd;   

        // arrays to save the rotations for kernel mergeRotate:
        rocblas_int* dcount = splits;
        S* c = vecs + 3 * n;
        S* s = vecs + 4 * n;

        rocblas_int i = pin;
        while(i < pout)
        {
            rocblas_int count = 0;
            rocblas_int map = pers[i];
            vald = vals[i];
            valz = zz[map];
            
            if(abs(p * valz) <= tol)
            {
                // deflate 'vald' because of zero component in Z
                nf++;
                evf[pout - nf] = vald;
                idrf[pout - nf] = map;
                dcount[i] = 0;
                i++;
            }
            else
            {
                // otherwise, 'vald' is not deflated and will be part of secular equation
                evr[pin + nr] = vald;
                idrf[pin + nr] = -(map + 1);
                rocblas_int oldi = i;
                
                // now, analyze the sequence of values close to 'vald', if any,  and deflate them
                i++;
                while(i < pout)
                { 
                    rocblas_int mapt = pers[i];
                    S valdt = vals[i];
                    S valzt = zz[mapt];

                    if(abs(vald - valdt) <= tol)
                    {
                        // deflate 'valdt' because is same as 'vald' for the given tolerance
                        nf++;
                        evf[pout - nf] = valdt; 
                        idrf[pout - nf] = mapt;
                        dcount[i] = 0;
                        i++;
                        
                        // find rotation to zero-out component of Z if necessary
                        if(abs(p * valzt) > tol)
                        {
                            S cc, ss, rr;
                            lartg(valz, valzt, cc, ss, rr);            
                            valz = rr;

                            // save the rotation encoded for mergeRotate
                            count++;
                            c[mapt] = cc;
                            s[mapt] = ss;
                        }
                    }
                    else
                        break;
                }
                ztmp[pin + nr] = valz;
                dcount[oldi] = count;
                nr++;
            }
        }
        
        // save 'nr' and 'nf'
        shmaxz[tx] = nr;
        shmaxd[tx] = nf;
        nrs[tx] = nr;
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPARE_KERNEL prepares the components for the secular equations of every 
    pair of sub-blocks that need to be merged.
        - Call this kernel with batch_count groups in z, and as many groups as needed in 
          x and y to cover the n rows and columns **/
template <typename S>
ROCSOLVER_KERNEL void 
stedc_mergePrepare_kernel(const rocblas_int levs,
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
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int dimr = hipGridDim_x * hipBlockDim_x;
    rocblas_int dimc = hipGridDim_y * hipBlockDim_y;
    // row id
    rocblas_int rid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // column/vector id
    rocblas_int cid = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    // select batch instance to work with
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    rocblas_int* nrs = pers + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (3 * n);
    // roots of secular equations
    S* evr = z + n;
    S* evf = z + 2*n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);
    S* ztmp = vecs + 2 * n;

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    for(int jj = cid; jj < n; jj += dimc)
    {
        // column 'jj' belongs to sub-block 'bx' and thus forms part of 
        // the new sub-block 'nbx'
        rocblas_int bx = bisearch(jj, ps, blks, false, false) - 1;
        rocblas_int nbx = bx / dm2;
        
        // the new sub-block starts at 'pin', the middle point is 'pmid', and
        // it ends at 'pout'. Element 'p' is found at middle point
        rocblas_int tmp = nbx * dm2;
        rocblas_int pin = ps[tmp];
        rocblas_int pmid = ps[tmp + dm];
        tmp += dm2;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;
        S p = 2 * E[pmid - 1];
        rocblas_int nr = nrs[nbx * dm2];  // number of non-deflated values in sub-block            
        rocblas_int j = jj - pin;

        if(j < nr)
        {
            S* tmpd = temps + pin * n;
            S* ev = evr + pin;
            S* Z = z + pin;

            // if 'p' is negative, the values are copied as negative in reverse order
            // as required by the secular equation solvers
            bool pneg = (p < 0);
            rocblas_int sig = pneg ? -1 : 1;
            rocblas_int start = pneg ? nr - 1 : 0;

            for(int i = rid; i < nr; i += dimr)
            {
                int id = start + sig * i;
                tmpd[i + j * n] = sig * ev[id];
                if(j == 0)
                    Z[i] = ztmp[id + pin];
            }
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEROTATE_KERNEL performs rotation of vectors corresponding to deflations
        - Call this kernel with batch_count groups in y, and n (matrix size) groups in x.
        - Each group will deal with one deflation group, groups that don't correspond to
          a deflation group will do nothing **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeRotate_kernel(const rocblas_int levs,
                             const rocblas_int blks,
                             const rocblas_int k,
                             const rocblas_int n,
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

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (3 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);

    rocblas_int* map = pers;         
    rocblas_int* dcounts = splits; 
    S* cc = vecs + 3 * n;
    S* ss = vecs + 4 * n; 

    constexpr int regs = 16;
    const int chunk_width = regs * hipBlockDim_x;
    const int n_chunks    = (n - 1) / chunk_width + 1;
    S bval[regs];
    S tval[regs];

    rocblas_int dgs = hipBlockIdx_x;
    rocblas_int dcnt = dcounts[dgs];
    if (dcnt)
    {
        rocblas_int base = map[dgs];
        S* Cbase = C + base * ldc;

        for (int chunk = 0; chunk < n_chunks; chunk++) 
        {
            for(int i = 0; i < regs; i++) 
            {
                int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                if (x < n) 
                    bval[i] = Cbase[x];
            }

            for (int dn = 0; dn < dcnt; dn++) 
            {
                rocblas_int top = map[dgs + dn + 1];
                S c = cc[top];
                S s = ss[top];
                S* Ctop = C + top * ldc;

                for(int i = 0; i < regs; i++) 
                {
                    int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                    if(x < n)
                        tval[i] = Ctop[x];
                }

                for (int i = 0; i < regs; i++) 
                {
                    S valf = bval[i];
                    S valg = tval[i];
                    bval[i] = valf * c - valg * s;
                    tval[i] = valf * s + valg * c; 
                }

                for(int i = 0; i < regs; i++) 
                {
                    int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                    if(x < n)
                        Ctop[x] = tval[i];
                }
                __syncthreads();
            }

            for(int i = 0; i < regs; i++) 
            {
                int x = chunk * chunk_width + i * hipBlockDim_x + hipThreadIdx_x;
                if (x < n) 
                    Cbase[x] = bval[i];
            }
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVALUES_KERNEL solves the secular equation for every value of every pair of 
    sub-blocks that need to be merged, and re-scales vector z accordingly.
        - Call this kernel with batch_count groups in y, and as many groups in x as needed
          to cover the n values of the matrix.
        - Each thread will deal with one value.  
        - Size of groups is set to STEDC_BDIM.**/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
stedc_mergeValues_kernel(const rocblas_int levs,
                       const rocblas_int blks,
                       const rocblas_int k,
                       const rocblas_int n,
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
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    rocblas_int* nrs = pers + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (3 * n);
    // roots of secular equations
    S* evs = z + 2*n;
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
        rocblas_int bx = bisearch(tx, ps, blks, false, false) - 1;
        rocblas_int nbx = bx / dm2;

        // the new sub-block starts at 'pin', the middle point is 'pmid', and
        // it ends at 'pout'. Element 'p' is found at middle point
        rocblas_int tmp = nbx * dm2;
        rocblas_int pin = ps[tmp];
        rocblas_int pmid = ps[tmp + dm];
        tmp += dm2;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;
        S p = 2 * E[pmid - 1];
        rocblas_int nr = nrs[nbx * dm2];  // number of non-deflated values in sub-block

        // 1. solve secular equation for every non-deflated value
        // ------------------------------------------------------------------
        rocblas_int linfo;
        
        if(idd[tx] < 0)
        {
#if defined(ROCSOLVER_USE_REFERENCE_SECULAR_EQUATIONS_SOLVER)
            linfo = slaed4(nr, tx - pin, temps + tx * n, z + pin, std::abs(p), evs[tx]);
#else
            if(tx - pin == nr - 1)
                linfo = seq_solve_ext(nr, temps + tx * n, z + pin, std::abs(p), evs[tx], eps,
                                      ssfmin, ssfmax);
            else
                linfo = seq_solve(nr, temps + tx * n, z + pin, std::abs(p), evs[tx], eps,
                                  ssfmin, ssfmax);
#endif
            if(p < 0)
                evs[tx] *= -1; 
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEREINSERT_KERNEL combines and sort the new eigenvalues with the deflated values
        - Call this kernel with batch_count groups in y, and as many groups in x as needed
          to cover the n values of the matrix.
        - Each thread will deal with one value.
        - Size of groups is set to STEDC_BDIM.**/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeReinsert_kernel(const rocblas_int levs,
                              const rocblas_int blks,
                              const rocblas_int k,
                              const rocblas_int n,
                              S* DD,
                              const rocblas_stride strideD,
                              S* EE,
                              const rocblas_stride strideE,
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
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    rocblas_int* nrs = pers + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (3 * n);
    // roots of secular equations
    S* ev = z + n;
    S* evs = z + 2*n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // work with all the values (items) in parallel
    for(rocblas_int j = tid; j < n; j += totdim)
    {
        // item 'j' belongs to sub-block 'bx' and thus form vector of 
        // the new sub-block 'nbx'
        rocblas_int bx = bisearch(j, ps, blks, false, false) - 1;
        rocblas_int nbx = bx / dm2;
        
        // the new sub-block starts at 'pin', the middle point is 'pmid', and
        // it ends at 'pout'. Element 'p' is found at middle point
        rocblas_int tmp = nbx * dm2;
        rocblas_int pin = ps[tmp];
        rocblas_int pmid = ps[tmp + dm];
        tmp += dm2;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;
        S p = 2 * E[pmid - 1];
        rocblas_int nr = nrs[nbx * dm2];  // number of non-deflated values in sub-block

        // 1. re-insert deflated values to keep new sub-blocks ordered
        // -----------------------------------------------------------------------
        rocblas_int* ord = splits;
        rocblas_int nf = pout - pin - nr;   // number of deflated values 
        
        // the position where the item 'j' will end up in the ordered array is 'pos'
        S val = evs[j];
        rocblas_int pos1 = (j < nr + pin) ? bisearch(val, evs + nr + pin, nf, true, true)
                                          : bisearch(val, evs + pin, nr, false, (p < 0));
        rocblas_int pos2 = (j < nr + pin) ? (p < 0 ? pin + nr - 1 - j : j - pin)   
                                          : pout - j - 1;
        rocblas_int pos = pos1 + pos2 + pin;

        // get merged ordered array 'ev' and permutation map 'ord'
        D[pos] = val;

        rocblas_int ind = idd[j];
        if(ind < 0)
            ord[pos] = -(j + 1);
        else
            ord[pos] = ind;
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGERESCALE_KERNEL reconstructs perturbed vector Z of the rank-1 system.
        - Call this kernel with batch_count groups in z, blks groups in y and n groups in x.
        - Each group will deal with one row of Z corresponding to each merge.
        - Size of groups is set to STEDC_BDIM.**/
template <bool USEGEMM, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeRescale_kernel(const rocblas_int levs,
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
    rocblas_int bid = hipBlockIdx_z;
    // row id
    rocblas_int i = hipBlockIdx_x;
    // sub-block id
    rocblas_int nbx = hipBlockIdx_y;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    rocblas_int* nrs = pers + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (3 * n);
    // roots of secular equations
    S* ev = z + n;
    S* evs = z + 2*n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // temporary arrays in shared memory
    // used to store temp values during the different reductions
    __shared__ S inrms[STEDC_BDIM];

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    if(nbx < blks / dm2)
    {
        // the new sub-block starts at 'pin', the middle point is 'pmid', and
        // it ends at 'pout'. Element 'p' is found at middle point
        rocblas_int tmp = nbx * dm2;
        rocblas_int pin = ps[tmp];
        rocblas_int pmid = ps[tmp + dm];
        tmp += dm2;
        rocblas_int pout = tmp < blks ? ps[tmp] : n;
        S p = 2 * E[pmid - 1];
        rocblas_int nr = nrs[nbx * dm2];  // number of non-deflated values in sub-block

        rocblas_int* id = idd + pin;
        S* evd = ev + pin;
        rocblas_int start = (p < 0) ? nr - 1 : 0;
        rocblas_int inc = (p < 0) ? -1 : 1;

        // 1. compute re-scaled vector Z of rank-1 perturbed system 
        // --------------------------------------------------------------------
        if(i < nr)
        {
            rocblas_int sgnz = (z[i + pin] < 0) ? -1 : 1;
            S dd = evd[start + inc * i];
            S mul = 1;

            for(int j = tidb; j < nr; j += dim)
            {
                S num = std::abs(temps[i + (j + pin) * n]);
                S den = (j == i) ? 1 : std::abs(dd - evd[start + inc * j]);
                mul *= num / den;
            }
            inrms[tidb] = mul;
            __syncthreads();

            // reduction (for the norms)
            for(int r = dim / 2; r > 0; r /= 2)
            {
                if(tidb < r)
                {
                    mul *= inrms[tidb + r];
                    inrms[tidb] = mul;
                }
                __syncthreads();
            }

            if(tidb == 0)
                z[i + pin] = sgnz * std::sqrt(mul);
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVECTORS_KERNEL computes vectors of the rank-1 system for
    every pair of sub-blocks that need to be merged.
        - Call this kernel with batch_count groups in y, and n groups in x.
        - Each group works with a column/vector.
        - Groups are size STEDC_BDIM **/
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
    // column/vector id
    rocblas_int j = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    rocblas_int* nrs = pers + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (3 * n);
    // roots of secular equations
    S* ev = z + n;
    S* evs = z + 2*n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    // temporary arrays in shared memory
    // used to store temp values during the different reductions
    __shared__ S inrms[STEDC_BDIM];

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // column 'j' belongs to sub-block 'bx' and thus form vector of 
    // the new sub-block 'nbx'
    rocblas_int bx = bisearch(j, ps, blks, false, false) - 1;
    rocblas_int nbx = bx / dm2;
        
    // the new sub-block starts at 'pin', the middle point is 'pmid', and
    // it ends at 'pout'. Element 'p' is found at middle point
    rocblas_int tmp = nbx * dm2;
    rocblas_int pin = ps[tmp];
    rocblas_int pmid = ps[tmp + dm];
    tmp += dm2;
    rocblas_int pout = tmp < blks ? ps[tmp] : n;
    S p = 2 * E[pmid - 1];
    rocblas_int nr = nrs[nbx * dm2];  // number of non-deflated values in sub-block

    rocblas_int* id = idd + pin;
    S* evd = ev + pin;
    rocblas_int start = (p < 0) ? nr - 1 : 0;
    rocblas_int inc = (p < 0) ? -1 : 1;

    // 1. compute vectors of rank-1 perturbed system and their norms
    // --------------------------------------------------------------------
    if(idd[j] < 0 && j < n)
    { 
        S tm, nrm = 0;
        for(int i = tidb; i < nr; i += dim)
        {
            S tot = z[i + pin] / temps[i + j * n];
            vecs[i + j * n] = tot;
            nrm += tot * tot;
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
        nrm = std::sqrt(inrms[0]);        

        // normalize
        for(int i = tidb; i < nr; i += dim)
            vecs[i + j * n] /= nrm;
    }

    /*if(!USEGEMM)
    {
        // Vectors should be updated at this point when not using external gemm.
        // TODO: the code needs to be revisited and adapted for this new implementation
        // of stedc. Performance of the internal gemm, and other gemm options (like batched
        // gemm) needs to be evaluated.
    }*/
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPGEMM1_KERNEL prepares the matrix of vectors of the rank-1 system for 
    the gemm to update eigenvectors (pad with zeros and insert 1 for deflated values).
        - Call this kernel with batch_count groups in y, and as many groups as needed in 
          x to cover the n columns 
        - Groups are size STEDC_BDIM **/
template <typename S> __launch_bounds__(STEDC_BDIM)
ROCSOLVER_KERNEL void stedc_mergePrepgemm1_kernel(const rocblas_int levs,
                                              const rocblas_int blks,
                                              const rocblas_int k,
                                              const rocblas_int n,
                                              S* EE,
                                              const rocblas_stride strideE,
                                              S* tmpzA,
                                              S* vecsA,
                                              rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int dim = hipGridDim_x * hipBlockDim_x;
    rocblas_int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    rocblas_int* nrs = pers + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (3 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    for(int j = tid; j < n; j += dim)
    {
        rocblas_int i = splits[j];
        if(i >= 0)
            temps[i + j * n] = 1;
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPGEMM_KERNEL prepares the matrix of vectors of the rank-1 system for 
    the gemm to update eigenvectors (pad with zeros and permutate rows and columns).
        - Call this kernel with batch_count groups in y, and n groups in x 
        - Groups are size STEDC_BDIM **/
template <typename S> __launch_bounds__(STEDC_BDIM)
ROCSOLVER_KERNEL void stedc_mergePrepgemm_kernel(const rocblas_int levs,
                                              const rocblas_int blks,
                                              const rocblas_int k,
                                              const rocblas_int n,
                                              S* EE,
                                              const rocblas_stride strideE,
                                              S* tmpzA,
                                              S* vecsA,
                                              rocblas_int* splitsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_y;
    // column vector id
    rocblas_int j = hipBlockIdx_x;
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;

    // select batch instance to work with
    S* E = EE + bid * strideE;

    // temporary arrays in global memory
    rocblas_int* splits = splitsA + bid * (5 * n + blks);
    // the sub-blocks sizes
    rocblas_int* ns = splits + n;
    // the sub-blocks initial positions
    rocblas_int* ps = ns + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = ps + n;
    // container of permutations when solving the secular eqns
    rocblas_int* pers = idd + n;
    rocblas_int* nrs = pers + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (3 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    // temp values during the merges
    S* temps = vecs + (n * n);

    rocblas_int dm = 1 << k;
    rocblas_int dm2 = dm << 1;

    // column 'j' belongs to sub-block 'bx' and thus form vector of
    // the new sub-block 'nbx'
    rocblas_int bx = bisearch(j, ps, blks, false, false) - 1;
    rocblas_int nbx = bx / dm2;

    // the new sub-block starts at 'pin', the middle point is 'pmid', and
    // it ends at 'pout'. Element 'p' is found at middle point
    rocblas_int tmp = nbx * dm2;
    rocblas_int pin = ps[tmp];
    rocblas_int pmid = ps[tmp + dm];
    tmp += dm2;
    rocblas_int pout = tmp < blks ? ps[tmp] : n;
    S p = 2 * E[pmid - 1];
    rocblas_int nr = nrs[nbx * dm2];  // number of non-deflated values in sub-block

    rocblas_int start = (p < 0) ? pin + nr - 1 : pin;
    rocblas_int inc = (p < 0) ? -1 : 1;

    // 1. put vectors in padded matrix 'temps' to use external gemm for the update
    // -----------------------------------------------------------------------
    rocblas_int ind = splits[j];
    
    if(ind < 0)
    {
        for(int i = tid; i < nr; i += dim)
        {
            // read rank-1 vector value from 'vecs' (this permutates columns)
            rocblas_int jv = -(ind + 1);     
            S val = vecs[i + jv * n];

            // write in final position in 'temps' (this permutates rows)
            rocblas_int it = -(idd[start + inc * i] + 1);
            temps[it + j * n] = val;
        }
    }
}


//--------------------------------------------------------------------------------------//
/** STEDC_MERGEUPDATE_KERNEL updates vectors after a merge is done. 
    (simply copy results from temporary arrays into V)
        - Call this kernel with batch_count groups in z, and as many groups as needed in 
          x and y to cover the n rows and columns **/
template <typename S>
ROCSOLVER_KERNEL void 
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
                             S* vecsA)
{
    // threads and groups indices
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int dimr = hipGridDim_x * hipBlockDim_x;
    rocblas_int dimc = hipGridDim_y * hipBlockDim_y;
    // row id
    rocblas_int rid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // column/vector id
    rocblas_int cid = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    // select batch instance to work with
    S* C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;

    // temporary arrays in global memory
    // updated eigenvalues after merge
    S* evs = tmpzA + bid * (3 * n);
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);

    for(int j = cid; j < n; j += dimc)
    {
        for(int i = rid; i < n; i += dimr)
            C[i + j * ldc] = vecs[i + j * n];
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

    return levels;
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
        // find number of sub-blocks 
        rocblas_int levs = stedc_num_levels(n);
        rocblas_int blks = 1 << levs;

        // requirements for solver of small independent blocks
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, size_work_stack);

        // extra requirements for original eigenvectors of small independent blocks
        if(evect != rocblas_evect_tridiagonal)
            *size_tempvect = sizeof(S) * (n * n) * batch_count;
        else
            *size_tempvect = 0;
        *size_tempgemm = sizeof(S) * 2 * (n * n) * batch_count;

        // blocks for batched GEMM are at least 8 x 8
        *size_workArr = (n / 8) * sizeof(S*) * 3;

        // size for split blocks and sub-blocks positions
        *size_splits_map = sizeof(rocblas_int) * (5 * n + blks) * batch_count;

        // size for temporary diagonal and rank-1 modif vector
        *size_tmpz = sizeof(S) * (3 * n) * batch_count;
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
//int ttt = atoi(getenv("DEBUG"));
//bool print_debug = (ttt == 1);
//ttt = atoi(getenv("TIMES"));
//bool print_times = (ttt == 1);
bool print_debug = false;
bool print_times = true;

hipEvent_t setup_events[4];
for(int i = 0; i < 4; i++)
    HIP_CHECK(hipEventCreate(&setup_events[i]));

if(print_debug)
{
printf("\n");
print_device_matrix(std::cout,"D in",1,n,D,1);
print_device_matrix(std::cout,"E in",1,n-1,E,1);
}

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

HIP_CHECK(hipEventRecord(setup_events[0], stream));
        ROCSOLVER_LAUNCH_KERNEL(init_ident<S>, dim3(groupsn, groupsn, batch_count), dim3(BS2, BS2),
                                0, stream, n, n, V, 0, ldv, strideV);

        // 1. divide phase
        //-----------------------------
        rocblas_int groups = (batch_count - 1) / STEDC_BDIM + 1;

HIP_CHECK(hipEventRecord(setup_events[1], stream));
        ROCSOLVER_LAUNCH_KERNEL((stedc_divide_kernel<S>),
                                dim3(groups), dim3(STEDC_BDIM), 0, stream, levs, blks, n, D + shiftD,
                                strideD, E + shiftE, strideE, batch_count, splits);

        // 2. solve phase
        //-----------------------------
HIP_CHECK(hipEventRecord(setup_events[2], stream));
        ROCSOLVER_LAUNCH_KERNEL((stedc_solve_kernel<S>),
                                dim3(blks, batch_count), dim3(WAVEFRONT), 0, stream, levs, blks, 
                                n, D + shiftD, strideD, E + shiftE, strideE, 
                                V, 0, ldv, strideV, info, (S*)work_stack, splits, 
                                eps, ssfmin, ssfmax);

HIP_CHECK(hipEventRecord(setup_events[3], stream));

        HIP_CHECK(hipStreamSynchronize(stream));

        float elapsed[3];
        for(int i = 0; i < 3; i++)
            HIP_CHECK(hipEventElapsedTime(&elapsed[i], setup_events[i], setup_events[i+1]));
        for(int i = 0; i < 4; i++)
            HIP_CHECK(hipEventDestroy(setup_events[i]));

if(print_times)
{
printf("\n\tinit_ident         : %f\n"
       "\tstedc_divide_kernel: %f\n"
       "\tstedc_solve_kernel : %f\n\n",
       elapsed[0], elapsed[1], elapsed[2]);
}

if(print_debug)
{
print_device_matrix(std::cout,"ns",1,n,splits+n,1);
print_device_matrix(std::cout,"ps",1,n,splits+2*n,1);
//print_device_matrix(std::cout,"D at leaves",1,n,D,1);
//print_device_matrix(std::cout,"E at leaves",1,n-1,E,1);
//print_device_matrix(std::cout,"V at leaves",n,n,V,ldv);
}

        // 3. merge phase
        //----------------
        size_t lmemsize = sizeof(S) * blks;
        rocblas_int numgrps = (n - 1) / STEDC_BDIM + 1;
        rocblas_int numthds = ((blks - 1) / WAVEFRONT + 1) * WAVEFRONT;

        // launch merge for level k
        for(rocblas_int k = 0; k < levs; ++k) ////////////////////////////////////////////////////////////////////////////// k < levs
        {

hipEvent_t merge_events[14];            
for(int i = 0; i < 14; i++)
    HIP_CHECK(hipEventCreate(&merge_events[i]));


            // a. prepare secular equations
if(print_times || print_debug)
{
printf("------------------------------------------\n");
printf("     start merge at level k = %d\n",k);
printf("------------------------------------------\n\n");
}
if(print_debug)
{
print_device_matrix(std::cout,"values to be merge-sorted",1,n,D,1);
}

HIP_CHECK(hipEventRecord(merge_events[0], stream));
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeSort_kernel<S>), dim3(numgrps, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, D + shiftD, strideD,
                                    V, 0, ldv, strideV, tmpz, tempgemm, splits);
if(print_debug)
{
printf("after mergeSort:\n");
printf("---------------------\n");
print_device_matrix(std::cout,"ids of the sort",1,n,splits+4*n,1);
print_device_matrix(std::cout,"z after sort",1,n,tempgemm,1);
print_device_matrix(std::cout,"values after sort",1,n,tempgemm+n,1);
}
            
HIP_CHECK(hipEventRecord(merge_events[1], stream));
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeDeflate_kernel<S>), dim3(1, batch_count),
                                    dim3(numthds), lmemsize, stream, levs, blks, k, n, E + shiftE, strideE,
                                    tmpz, tempgemm, splits, eps);
if(print_debug)
{
printf("after mergeDeflate:\n");
printf("---------------------\n");
print_device_matrix(std::cout,"size of non-deflated",1,blks,splits+5*n,1);
print_device_matrix(std::cout,"ids after deflation",1,n,splits+3*n,1);
print_device_matrix(std::cout,"non deflated values",1,n,tmpz+n,1);
print_device_matrix(std::cout,"deflated values",1,n,tmpz+2*n,1);
print_device_matrix(std::cout,"dcount of rotations",1,n,splits,1);
print_device_matrix(std::cout,"vecs after deflate",n,n,tempgemm,n);
print_device_matrix(std::cout,"temps after deflate",n,n,tempgemm+n*n,n);
}

HIP_CHECK(hipEventRecord(merge_events[2], stream));
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_kernel<S>), dim3(groupsn, groupsn, batch_count), 
                                    dim3(BS2,BS2), 0, stream, levs, blks, k, n, E + shiftE, strideE,
                                    tmpz, tempgemm, splits, eps);

if(print_debug)
{
printf("after mergePrepare:\n");
printf("---------------------\n");
print_device_matrix(std::cout,"vecs after prepare",n,n,tempgemm,n);
print_device_matrix(std::cout,"temps after prepare",n,n,tempgemm+n*n,n);
print_device_matrix(std::cout,"Z for secular eqns",1,n,tmpz,1);
}



HIP_CHECK(hipEventRecord(merge_events[3], stream));
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeRotate_kernel<S>), dim3(n, batch_count),
                                    dim3(STEDC_BDIM),
                                    0, stream, levs, blks, k, n,
                                    V, 0, ldv, strideV, tmpz, tempgemm, splits);
if(print_debug)
{
printf("after mergeRotate:\n");
printf("---------------------\n");
print_device_matrix(std::cout,"V after rotate",n,n,V,ldv);            
}

            // b. solve secular eq to find merged eigenvalues
HIP_CHECK(hipEventRecord(merge_events[4], stream));
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_kernel<S>), dim3(numgrps, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, E + shiftE, strideE,
                                    tmpz, tempgemm, splits, eps, ssfmin, ssfmax);

HIP_CHECK(hipEventRecord(merge_events[5], stream));
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeReinsert_kernel<S>), dim3(numgrps, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, levs, blks, k, n, D + shiftD, strideD, E + shiftE, strideE,
                                    tmpz, tempgemm, splits);

if(print_debug)
{
printf("after mergeValues:\n");
printf("---------------------\n");
print_device_matrix(std::cout,"orig values to secular eqns",1,n,tmpz+n,1);
print_device_matrix(std::cout,"new values from secular eqns",1,n,tmpz+2*n,1);
print_device_matrix(std::cout,"values with deflated re-indserted in order",1,n,D,1);
print_device_matrix(std::cout,"final order",1,n,splits,1);
print_device_matrix(std::cout,"vecs after values",n,n,tempgemm,n);
print_device_matrix(std::cout,"temps after values",n,n,tempgemm+n*n,n);
}

            // c. find merged eigenvectors
HIP_CHECK(hipEventRecord(merge_events[6], stream));
            ROCSOLVER_LAUNCH_KERNEL(
                (stedc_mergeRescale_kernel<STEDC_EXTERNAL_GEMM, S>),
                dim3(n, blks, batch_count), dim3(STEDC_BDIM), 0, stream,
                levs, blks, k, n, D + shiftD, strideD, E + shiftE, strideE, V, 0, ldv, strideV,
                tmpz, tempgemm, splits);        

HIP_CHECK(hipEventRecord(merge_events[7], stream));
            ROCSOLVER_LAUNCH_KERNEL(
                (stedc_mergeVectors_kernel<STEDC_EXTERNAL_GEMM, S>),
                dim3(n, batch_count), dim3(STEDC_BDIM), 0, stream, 
                levs, blks, k, n, D + shiftD, strideD, E + shiftE, strideE, V, 0, ldv, strideV, 
                tmpz, tempgemm, splits);

if(print_debug)
{
printf("after mergeVectors:\n");
printf("---------------------\n");
print_device_matrix(std::cout,"vecs after vectors",n,n,tempgemm,n);
print_device_matrix(std::cout,"temps after vectors",n,n,tempgemm+n*n,n);
}

            if(STEDC_EXTERNAL_GEMM)
            {
                // using external gemms with padded matrices to do the vector update
                // One single full gemm of size n x n x n merges all the blocks in the level
                // TODO: using macro STEDC_EXTERNAL_GEMM = true for now. In the future we can pass
                // STEDC_EXTERNAL_GEMM at run time to switch between internal vector updates and
                // external gemm based updates.

HIP_CHECK(hipEventRecord(merge_events[8], stream));
                HIP_CHECK(hipMemsetAsync((void*)(tempgemm+n*n), 0, sizeof(S) * n * n, stream));

HIP_CHECK(hipEventRecord(merge_events[9], stream));
                ROCSOLVER_LAUNCH_KERNEL(stedc_mergePrepgemm1_kernel<S>,
                dim3(numgrps, batch_count), dim3(STEDC_BDIM), 0, stream,
                levs, blks, k, n, E + shiftE, strideE, tmpz, tempgemm, splits);

if(print_debug)
{
print_device_matrix(std::cout,"temps after prepgem1",n,n,tempgemm+n*n,n);
}

HIP_CHECK(hipEventRecord(merge_events[10], stream));
                ROCSOLVER_LAUNCH_KERNEL(stedc_mergePrepgemm_kernel<S>,
                dim3(n, batch_count), dim3(STEDC_BDIM), 0, stream,
                levs, blks, k, n, E + shiftE, strideE, tmpz, tempgemm, splits);

if(print_debug)
{
print_device_matrix(std::cout,"temps after prepgem",n,n,tempgemm+n*n,n);
}


HIP_CHECK(hipEventRecord(merge_events[11], stream));

                if(n <= 1024 || batch_count > 1)
                {
                    rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n,
                                   &one, V, 0, ldv, strideV, tempgemm, n * n, n, 2 * n * n, &zero,
                                   tempgemm, 0, n, 2 * n * n, batch_count, workArr);
                }
                else
                {

                    HIP_CHECK(hipMemsetAsync((void*)tempgemm, 0, n*n*sizeof(S), stream));
                    // HIP_CHECK(hipMemset((void*)tempgemm, 0, n*n*sizeof(S)));

                    rocblas_int lvl = levs - k - 1;
                    rocblas_int nb = 1 << lvl;
                    std::vector<rocblas_int> ns(nb);
                    ns[0] = n;
                    for(int i = 0; i < lvl; ++i)
                    {
                        for(int j = (1 << i); j > 0; --j)
                        {
                            auto t = ns[j - 1];
                            auto t2 = t / 2;
                            ns[j * 2 - 1] = (2 * t2 < t) ? t2 + 1 : t2;
                            ns[j * 2 - 2] = t2;
                        }
                    }
                    if(std::all_of(ns.begin(), ns.end(), [&](rocblas_int v) { return v == ns[0]; }))
                    {
                        rocsolver_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                                       ns[0], ns[0], ns[0], &one, V, 0, ldv, ns[0] * ldv + ns[0],
                                       tempgemm, n * n, n, ns[0] * n + ns[0], &zero, tempgemm, 0, n,
                                       ns[0] * n + ns[0], nb, workArr);
                    }
                    else
                    {
                        // there can only be 2 block sizes: ns[0] and ns[0]+1
                        std::array<std::vector<rocblas_int>, 2> uniform_batch;
                        uniform_batch[0].reserve(nb);
                        uniform_batch[1].reserve(nb);
                        for(rocblas_int i = 0, ps = 0; i < nb; ps += ns[i++])
                            uniform_batch[ns[i] != ns[0]].push_back(ps);
                        for(rocblas_int i = 0, nsb = ns[0]; i < 2; ++i, ++nsb)
                        {
                            auto& b = uniform_batch[i];
                            auto nbb = b.size();
                            std::vector<S*> hABC(nbb * 3);
                            for(size_t j = 0; j < nbb; ++j)
                            {
                                auto ps = b[j];
                                hABC[j] = V + ps * ldv + ps;
                                hABC[j + nbb] = tempgemm + n * n + ps * n + ps;
                                hABC[j + 2 * nbb] = tempgemm + ps * n + ps;
                            }
                            HIP_CHECK(hipMemcpy(workArr, hABC.data(), 3 * nbb * sizeof(S*),
                                                hipMemcpyHostToDevice));
                            rocsolver_gemm<S, rocblas_int, S* const*, S* const*, S* const*>(
                                handle, rocblas_operation_none, rocblas_operation_none, nsb, nsb,
                                nsb, &one, workArr, 0, ldv, 0, workArr + nbb, 0, n, 0, &zero,
                                workArr + 2 * nbb, 0, n, 0, nbb, nullptr);
                        }
                    }
                }


if(print_debug)
{
print_device_matrix(std::cout,"new vectors",n,n,tempgemm,n);
}
            }

            // d. update level
HIP_CHECK(hipEventRecord(merge_events[12], stream));
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeUpdate_kernel<S>),
                                    dim3(groupsn, groupsn, batch_count), dim3(BS2,BS2), 0, stream, 
                                    levs, blks, k, n, D + shiftD, strideD,
                                    V, 0, ldv, strideV, tmpz, tempgemm);

HIP_CHECK(hipEventRecord(merge_events[13], stream));

            HIP_CHECK(hipStreamSynchronize(stream));
            float merge_elapsed[13];

            for(int i = 0; i < 13; i++)
                HIP_CHECK(hipEventElapsedTime(&merge_elapsed[i], merge_events[i], merge_events[i+1]));
            for(int i = 0; i < 14; i++)
                HIP_CHECK(hipEventDestroy(merge_events[i]));

if(print_times)
{
            printf("\tmergeSort          : %f\n"
                   "\tmergeDeflate       : %f\n"
                   "\tmergePrepare       : %f\n"
                   "\tmergeRotate        : %f\n"
                   "\tmergeValues        : %f\n"
                   "\tmergeReinsert      : %f\n"
                   "\tmergeRescale       : %f\n"
                   "\tmergeVectors       : %f\n"
                   "\tmemset             : %f\n" 
                   "\tmergePrepgemm1     : %f\n"
                   "\tmergePrepgemm      : %f\n" 
                   "\tGEMM               : %f\n"
                   "\tmergeUpdate        : %f\n\n",
                merge_elapsed[0], merge_elapsed[1], merge_elapsed[2], merge_elapsed[3], merge_elapsed[4], 
                merge_elapsed[5], merge_elapsed[6], merge_elapsed[7], merge_elapsed[8], merge_elapsed[9],
                merge_elapsed[10], merge_elapsed[11], merge_elapsed[12]);
            fflush(stdout);
}

if(print_debug)
{
//print_device_matrix(std::cout,"new D",1,n,D,1);
//print_device_matrix(std::cout,"new V",n,n,V,ldv);
}

        }

        // 4. Final update 
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

        rocblas_set_pointer_mode(handle, old_mode);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
