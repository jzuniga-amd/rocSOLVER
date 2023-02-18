/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
 * ***********************************************************************/

#pragma once

#include "rocblas.hpp"
#include "roclapack_getf2.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"

#include <cstdlib>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fstream>
#include <iostream>
#include <string>

/** Execute all permutations dictated by the panel factorization
    in parallel (concurrency by rows and columns) **/
template <typename T, typename U>
ROCSOLVER_KERNEL void getrf_row_permutate(const rocblas_int n,
                                          const rocblas_int offset,
                                          const rocblas_int blk,
                                          U AA,
                                          const rocblas_int shiftA,
                                          const rocblas_int lda,
                                          const rocblas_stride strideA,
                                          rocblas_int* pividx,
                                          const rocblas_stride stridePI)
{
    int id = hipBlockIdx_z;
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;
    int bdx = hipBlockDim_x;
    int j = hipBlockIdx_y * hipBlockDim_y + ty;
    if(j >= offset)
        j += blk;

    if(j < n)
    {
        // batch instance
        T* A = load_ptr_batch(AA, id, shiftA, strideA);
        rocblas_int* piv = pividx + id * stridePI;

        // shared mem for temporary values
        extern __shared__ double lmem[];
        T* temp = reinterpret_cast<T*>(lmem);

        // do permutations in parallel (each tx perform a row swap)
        rocblas_int idx1 = piv[tx];
        rocblas_int idx2 = piv[idx1];
        temp[tx + ty * bdx] = A[idx1 + j * lda];
        A[idx1 + j * lda] = A[idx2 + j * lda];
        __syncthreads();

        // copy temp results back to A
        A[tx + j * lda] = temp[tx + ty * bdx];
    }
}

/** This is the implementation of the factorization of the
    panel blocks in getrf **/
template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status getrf_panelLU(rocblas_handle handle,
                             const rocblas_int mm,
                             const rocblas_int nn,
                             const rocblas_int n,
                             U A,
                             const rocblas_int r_shiftA,
                             const rocblas_int lda,
                             const rocblas_stride strideA,
                             rocblas_int* ipiv,
                             const rocblas_int shiftP,
                             const rocblas_stride strideP,
                             rocblas_int* info,
                             const rocblas_int batch_count,
                             const bool pivot,
                             T* scalars,
                             void* work1,
                             void* work2,
                             void* work3,
                             void* work4,
                             const bool optim_mem,
                             T* pivotval,
                             rocblas_int* pivotidx,
                             const rocblas_int offset,
                             rocblas_int* permut_idx,
                             const rocblas_stride stridePI)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // constants to use when calling rocablas functions
    T one = 1; // constant 1 in host
    T minone = -1; // constant -1 in host

    // r_shiftA is the row where the panel-block starts,
    // the actual position of the panel-block in the matrix is:
    rocblas_int shiftA = r_shiftA + idx2D(0, offset, lda);

    rocblas_int blk = getrf_get_innerBlkSize<ISBATCHED, T>(mm, nn, pivot);
    rocblas_int jb;
    rocblas_int dimx, dimy, blocks, blocksy;
    dim3 grid, threads;
    size_t lmemsize;

    // Main loop
    for(rocblas_int k = 0; k < nn; k += blk)
    {
        jb = min(nn - k, blk); // number of columns/pivots in the inner block

        // factorize inner panel block
        rocsolver_getf2_template<ISBATCHED, T>(handle, mm - k, jb, A, shiftA + idx2D(k, k, lda),
                                               lda, strideA, ipiv, shiftP + k, strideP, info,
                                               batch_count, scalars, pivotval, pivotidx, pivot,
                                               offset + k, permut_idx, stridePI);
        if(pivot)
        {
            dimx = jb;
            dimy = 1024 / dimx;
            blocks = (n - jb - 1) / dimy + 1;
            grid = dim3(1, blocks, batch_count);
            threads = dim3(dimx, dimy, 1);
            lmemsize = dimx * dimy * sizeof(T);

            // swap rows
            ROCSOLVER_LAUNCH_KERNEL(getrf_row_permutate<T>, grid, threads, lmemsize, stream, n,
                                    offset + k, jb, A, r_shiftA + k, lda, strideA, permut_idx,
                                    stridePI);
        }

        // update trailing sub-block
        if(k + jb < nn)
        {
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, jb,
                nn - k - jb, A, shiftA + idx2D(k, k, lda), lda, strideA, A,
                shiftA + idx2D(k, k + jb, lda), lda, strideA, batch_count, optim_mem, work1, work2,
                work3, work4);

            if(k + jb < mm)
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, mm - k - jb,
                    nn - k - jb, jb, &minone, A, shiftA + idx2D(k + jb, k, lda), lda, strideA, A,
                    shiftA + idx2D(k, k + jb, lda), lda, strideA, &one, A,
                    shiftA + idx2D(k + jb, k + jb, lda), lda, strideA, batch_count, nullptr);
            /** This would be the call to the internal gemm, leaving it
                    commented here until we are sure it won't be needed **/
            /*dimx = std::min({mm - k - jb, (4096 / jb) / 2, 32});
                dimy = std::min({nn - k - jb, (4096 / jb) / 2, 32});
                blocks = (mm - k - jb - 1) / dimx + 1;
                blocksy = (nn - k - jb - 1) / dimy + 1;
                grid = dim3(blocks, blocksy, batch_count);
                threads = dim3(dimx, dimy, 1);
                lmemsize = jb * (dimx + dimy) * sizeof(T);
                hipLaunchKernelGGL(gemm_kernel<T>, grid, threads, lmemsize, stream, mm - k - jb,
                                   nn - k - jb, jb, A, shiftA + idx2D(k + jb, k, lda),
                                   shiftA + idx2D(k, k + jb, lda),
                                   shiftA + idx2D(k + jb, k + jb, lda), lda, strideA);*/
        }
    }

    return rocblas_status_success;
}

/** Return the sizes of the different workspace arrays **/
template <bool BATCHED, bool STRIDED, typename T>
void rocsolver_getrf_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const bool pivot,
                                   const rocblas_int batch_count,
                                   size_t* size_scalars,
                                   size_t* size_work1,
                                   size_t* size_work2,
                                   size_t* size_work3,
                                   size_t* size_work4,
                                   size_t* size_pivotval,
                                   size_t* size_pivotidx,
                                   size_t* size_iipiv,
                                   size_t* size_iinfo,
                                   bool* optim_mem)
{
    static constexpr bool ISBATCHED = BATCHED || STRIDED;

    // if quick return, no need of workspace
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_scalars = 0;
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_pivotval = 0;
        *size_pivotidx = 0;
        *size_iipiv = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }

    rocblas_int dim = min(m, n);
    rocblas_int blk = getrf_get_blksize<ISBATCHED, T>(dim, pivot);

    if(blk == 0)
    {
        // requirements for one single GETF2
        rocsolver_getf2_getMemorySize<ISBATCHED, T>(m, n, pivot, batch_count, size_scalars,
                                                    size_pivotval, size_pivotidx);
        *size_work1 = 0;
        *size_work2 = 0;
        *size_work3 = 0;
        *size_work4 = 0;
        *size_iipiv = 0;
        *size_iinfo = 0;
        *optim_mem = true;
        return;
    }
    else
    {
        // requirements for largest possible GETF2 for the sub blocks
        // (largest block panel dimension is 512)
        rocsolver_getf2_getMemorySize<ISBATCHED, T>(
            m, min(dim, 512), pivot, batch_count, size_scalars, size_pivotval, size_pivotidx, true);

        // extra workspace to store info about singularity and pivots of sub blocks
        *size_iinfo = sizeof(rocblas_int) * batch_count;
        *size_iipiv = pivot ? m * sizeof(rocblas_int) * batch_count : 0;

        // extra workspace for calling largest possible TRSM
        rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_left, rocblas_operation_none,
                                                min(dim, 512), n, batch_count, size_work1,
                                                size_work2, size_work3, size_work4, optim_mem, true);
        if(!pivot)
        {
            size_t w1, w2, w3, w4;
            rocsolver_trsm_mem<BATCHED, STRIDED, T>(rocblas_side_right, rocblas_operation_none, m,
                                                    min(dim, 512), batch_count, &w1, &w2, &w3, &w4,
                                                    optim_mem, true);
            *size_work1 = std::max(*size_work1, w1);
            *size_work2 = std::max(*size_work2, w2);
            *size_work3 = std::max(*size_work3, w3);
            *size_work4 = std::max(*size_work4, w4);
        }
    }
}

template <bool BATCHED, bool STRIDED, typename T, typename U>
rocblas_status rocsolver_getrf_template(rocblas_handle handle,
                                        const rocblas_int m,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* ipiv,
                                        const rocblas_int shiftP,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* scalars,
                                        void* work1,
                                        void* work2,
                                        void* work3,
                                        void* work4,
                                        T* pivotval,
                                        rocblas_int* pivotidx,
                                        rocblas_int* iipiv,
                                        rocblas_int* iinfo,
                                        const bool optim_mem,
                                        const bool pivot)
{
    ROCSOLVER_ENTER("getrf", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "shiftP:", shiftP,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    static constexpr bool ISBATCHED = BATCHED || STRIDED;
    rocblas_int dim = min(m, n);
    rocblas_int blocks, blocksy;
    dim3 grid, threads;

    // quick return if no dimensions
    if(m == 0 || n == 0)
    {
        blocks = (batch_count - 1) / BS1 + 1;
        grid = dim3(blocks, 1, 1);
        threads = dim3(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, grid, threads, 0, stream, info, batch_count, 0);
        return rocblas_status_success;
    }

    // size of outer blocks
    rocblas_int blk = getrf_get_blksize<ISBATCHED, T>(dim, pivot);
printf("\n\n Block size is: %d\n\n",blk);

    if(blk == 0)
        return rocsolver_getf2_template<ISBATCHED, T>(handle, m, n, A, shiftA, lda, strideA, ipiv,
                                                      shiftP, strideP, info, batch_count, scalars,
                                                      pivotval, pivotidx, pivot);

    bool printing = false;
    const char* outfolder;
    if((outfolder = std::getenv("GETRF_OUT_FOLDER")) != nullptr)
        printing = true;

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    T one = 1;
    T minone = -1;

    rocblas_int jb, dimx, dimy;
    rocblas_int nextpiv, mm, nn;
    size_t lmemsize;
    rocblas_int j = 0;

    // in the npvt cases, panel determines whether the whole block-panel or only the
    // diagonal block is factorized
    bool panel = false;
    if(blk < 0)
    {
        panel = true;
        blk = -blk;
    }

    // MAIN LOOP
    rocblas_int iter = 0;
    for(rocblas_int j = 0; j < dim; j += blk)
    {
        jb = min(dim - j, blk);

        if(pivot || panel)
        {
            // factorize outer block panel
            getrf_panelLU<BATCHED, STRIDED, T>(handle, m - j, jb, n, A, shiftA + j, lda, strideA,
                                               ipiv, shiftP + j, strideP, info, batch_count, pivot,
                                               scalars, work1, work2, work3, work4, optim_mem,
                                               pivotval, pivotidx, j, iipiv, m);
        }
        else
        {
            // factorize only outer diagonal block
            getrf_panelLU<BATCHED, STRIDED, T>(handle, jb, jb, n, A, shiftA + j, lda, strideA, ipiv,
                                               shiftP + j, strideP, info, batch_count, pivot,
                                               scalars, work1, work2, work3, work4, optim_mem,
                                               pivotval, pivotidx, j, iipiv, m);

            // update remaining rows in outer panel
            rocsolver_trsm_upper<BATCHED, STRIDED, T>(
                handle, rocblas_side_right, rocblas_operation_none, rocblas_diagonal_non_unit,
                m - j - jb, jb, A, shiftA + idx2D(j, j, lda), lda, strideA, A,
                shiftA + idx2D(jb + j, j, lda), lda, strideA, batch_count, optim_mem, work1, work2,
                work3, work4);
        }

        if(printing)
        {
            std::string filename = fmt::format("{}/blkC_{}.txt", outfolder, iter);
            std::cout << fmt::format("Printing {} at j={}...", filename, j);

            std::ofstream file;
            file.open(filename);
            print_device_matrix(file, "Column block", m - j, jb, A + shiftA + idx2D(j, j, lda), lda);
            file.close();

            std::cout << "Done!" << std::endl;
        }

        // update trailing matrix
        nextpiv = j + jb; //position for the matrix update
        mm = m - nextpiv; //size for the matrix update
        nn = n - nextpiv; //size for the matrix update
        if(nextpiv < n)
        {
            rocsolver_trsm_lower<BATCHED, STRIDED, T>(
                handle, rocblas_side_left, rocblas_operation_none, rocblas_diagonal_unit, jb, nn, A,
                shiftA + idx2D(j, j, lda), lda, strideA, A, shiftA + idx2D(j, nextpiv, lda), lda,
                strideA, batch_count, optim_mem, work1, work2, work3, work4);

            if(printing)
            {
                std::string filename = fmt::format("{}/blkR_{}.txt", outfolder, iter);
                std::cout << fmt::format("Printing {} at j={}...", filename, j);

                std::ofstream file;
                file.open(filename);
                print_device_matrix(file, "Row block", jb, nn, A + shiftA + idx2D(j, nextpiv, lda),
                                    lda);
                file.close();

                std::cout << "Done!" << std::endl;
            }

            if(nextpiv < m)
            {
                rocblasCall_gemm<BATCHED, STRIDED, T>(
                    handle, rocblas_operation_none, rocblas_operation_none, mm, nn, jb, &minone, A,
                    shiftA + idx2D(nextpiv, j, lda), lda, strideA, A,
                    shiftA + idx2D(j, nextpiv, lda), lda, strideA, &one, A,
                    shiftA + idx2D(nextpiv, nextpiv, lda), lda, strideA, batch_count, nullptr);
                /** This would be the call to the internal gemm, leaving it
                        commented here until we are sure it won't be needed **/
                /*dimx = std::min({mm, (4096 / jb) / 2, 32});
                    dimy = std::min({nn, (4096 / jb) / 2, 32});
                    blocks = (mm - 1) / dimx + 1;
                    blocksy = (nn - 1) / dimy + 1;
                    grid = dim3(blocks, blocksy, batch_count);
                    threads = dim3(dimx, dimy, 1);
                    lmemsize = jb * (dimx + dimy) * sizeof(T);
                    hipLaunchKernelGGL(gemm_kernel<T>, grid, threads, lmemsize, stream, mm,
                                       nn, jb, A, shiftA + idx2D(nextpiv, j, lda),
                                       shiftA + idx2D(j, nextpiv, lda),
                                       shiftA + idx2D(nextpiv, nextpiv, lda), lda, strideA);*/

                if(printing)
                {
                    std::string filename = fmt::format("{}/trm_{}.txt", outfolder, iter);
                    std::cout << fmt::format("Printing {} at j={}...", filename, j);

                    std::ofstream file;
                    file.open(filename);
                    print_device_matrix(file, "Trailing matrix", mm, nn,
                                        A + shiftA + idx2D(nextpiv, nextpiv, lda), lda);
                    file.close();

                    std::cout << "Done!" << std::endl;
                }
            }
        }

        iter++;
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}
