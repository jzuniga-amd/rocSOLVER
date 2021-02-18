/* ************************************************************************
 * Copyright (c) 2020-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <bool STRIDED, typename T, typename S, typename U>
void syev_heev_checkBadArgs(const rocblas_handle handle,
                            const rocblas_evect evect,
                            const rocblas_fill uplo,
                            const rocblas_int n,
                            T dA,
                            const rocblas_int lda,
                            const rocblas_stride stA,
                            S dD,
                            const rocblas_stride stD,
                            S dE,
                            const rocblas_stride stE,
                            U dinfo,
                            const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, nullptr, evect, uplo, n, dA, lda, stA, dD,
                                              stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, rocblas_evect(-1), uplo, n, dA, lda,
                                              stA, dD, stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_value);
    EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, rocblas_fill_full, n, dA, lda,
                                              stA, dD, stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_value);

    // sizes (only check batch_count if applicable)
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA, lda, stA, dD,
                                                  stD, dE, stE, dinfo, -1),
                              rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, (T) nullptr, lda,
                                              stA, dD, stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA, lda, stA,
                                              (S) nullptr, stD, dE, stE, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA, lda, stA, dD,
                                              stD, (S) nullptr, stE, dinfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA, lda, stA, dD,
                                              stD, dE, stE, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, 0, (T) nullptr, lda,
                                              stA, (S) nullptr, stD, (S) nullptr, stE, dinfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    if(STRIDED)
        EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA, lda, stA, dD,
                                                  stD, dE, stE, (U) nullptr, 0),
                              rocblas_status_success);
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_syev_heev_bad_arg()
{
    using S = decltype(std::real(T{}));

    // safe arguments
    rocblas_local_handle handle;
    rocblas_evect evect = rocblas_evect_none;
    rocblas_fill uplo = rocblas_fill_lower;
    rocblas_int n = 1;
    rocblas_int lda = 1;
    rocblas_stride stA = 1;
    rocblas_stride stD = 1;
    rocblas_stride stE = 1;
    rocblas_int bc = 1;

    if(BATCHED)
    {
        // memory allocations
        device_batch_vector<T> dA(1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        syev_heev_checkBadArgs<STRIDED>(handle, evect, uplo, n, dA.data(), lda, stA, dD.data(), stD,
                                        dE.data(), stE, dinfo.data(), bc);
    }
    else
    {
        // memory allocations
        device_strided_batch_vector<T> dA(1, 1, 1, 1);
        device_strided_batch_vector<S> dD(1, 1, 1, 1);
        device_strided_batch_vector<S> dE(1, 1, 1, 1);
        device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, 1);
        CHECK_HIP_ERROR(dA.memcheck());
        CHECK_HIP_ERROR(dD.memcheck());
        CHECK_HIP_ERROR(dE.memcheck());
        CHECK_HIP_ERROR(dinfo.memcheck());

        // check bad arguments
        syev_heev_checkBadArgs<STRIDED>(handle, evect, uplo, n, dA.data(), lda, stA, dD.data(), stD,
                                        dE.data(), stE, dinfo.data(), bc);
    }
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void syev_heev_initData(const rocblas_handle handle,
                        const rocblas_evect evect,
                        const rocblas_int n,
                        Td& dA,
                        const rocblas_int lda,
                        const rocblas_int bc,
                        Th& hA,
                        std::vector<T>& A,
                        bool test = true)
{
    if(CPU)
    {
        rocblas_init<T>(hA, true);

        // scale A to avoid singularities
        for(rocblas_int b = 0; b < bc; ++b)
        {
            for(rocblas_int i = 0; i < n; i++)
            {
                for(rocblas_int j = 0; j < n; j++)
                {
                    if(i == j)
                        hA[b][i + j * lda] += 400;
                    else
                        hA[b][i + j * lda] -= 4;
                }
            }

            // make copy of original data to test vectors if required
            if(test && evect == rocblas_evect_original)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    for(rocblas_int j = 0; j < n; j++)
                        A[b * lda * n + i + j * lda] = hA[b][i + j * lda];
                }
            }
        }
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

template <bool STRIDED, typename T, typename Wd, typename Td, typename Id, typename Wh, typename Th, typename Ih>
void syev_heev_getError(const rocblas_handle handle,
                        const rocblas_evect evect,
                        const rocblas_fill uplo,
                        const rocblas_int n,
                        Wd& dA,
                        const rocblas_int lda,
                        const rocblas_stride stA,
                        Td& dD,
                        const rocblas_stride stD,
                        Td& dE,
                        const rocblas_stride stE,
                        Id& dinfo,
                        const rocblas_int bc,
                        Wh& hA,
                        Wh& hAres,
                        Th& hD,
                        Th& hDres,
                        Th& hE,
                        Ih& hinfo,
                        Ih& hinfoRes,
                        double* max_err)
{
    rocblas_int lwork = 5 * n;
    std::vector<T> A(lda * n * bc);

    // input data initialization
    syev_heev_initData<true, true, T>(handle, evect, n, dA, lda, bc, hA, A);

    // execute computations
    // CPU lapack
    for(rocblas_int b = 0; b < bc; ++b)
        cblas_syev_heev<T>(evect, uplo, n, hA[b], lda, hD[b], hE[b], lwork, hinfo[b]);

    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA.data(), lda, stA,
                                            dD.data(), stD, dE.data(), stE, dinfo.data(), bc));

    CHECK_HIP_ERROR(hDres.transfer_from(dD));
    CHECK_HIP_ERROR(hinfoRes.transfer_from(dinfo));
    if(evect == rocblas_evect_original)
        CHECK_HIP_ERROR(hAres.transfer_from(dA));

    // Check info for non-convergence
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
        if(hinfo[b][0] != hinfoRes[b][0])
            *max_err += 1;

    // (We expect the used input matrices to always converge. Testing
    // implicitly the equivalent non-converged matrix is very complicated and it boils
    // down to essentially run the algorithm again and until convergence is achieved).

    double err = 0;

    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(evect != rocblas_evect_original)
        {
            // only eigenvalues needed; can compare with LAPACK

            // error is ||hD - hDRes|| / ||hD||
            // using frobenius norm
            if(hinfo[b][0] == 0)
                err = norm_error('F', 1, n, 1, hD[b], hDres[b]);
            *max_err = err > *max_err ? err : *max_err;
        }
        else
        {
            // both eigenvalues and eigenvectors needed; need to implicitly test
            // eigenvectors due to non-uniqueness of eigenvectors under scaling
            if(hinfo[b][0] == 0)
            {
                // multiply A with each of the n eigenvectors and divide by corresponding
                // eigenvalues
                T alpha;
                T beta = 0;
                for(int j = 0; j < n; j++)
                {
                    alpha = T(1) / hDres[b][j];
                    cblas_symv_hemv(uplo, n, alpha, A.data() + b * lda * n, lda, hAres[b] + j * lda,
                                    1, beta, hA[b] + j * lda, 1);
                }

                // error is ||hA - hARes|| / ||hA||
                // using frobenius norm
                err = norm_error('F', n, n, lda, hA[b], hAres[b]);
                *max_err = err > *max_err ? err : *max_err;
            }
        }
    }
}

template <bool STRIDED, typename T, typename Wd, typename Td, typename Id, typename Wh, typename Th, typename Ih>
void syev_heev_getPerfData(const rocblas_handle handle,
                           const rocblas_evect evect,
                           const rocblas_fill uplo,
                           const rocblas_int n,
                           Wd& dA,
                           const rocblas_int lda,
                           const rocblas_stride stA,
                           Td& dD,
                           const rocblas_stride stD,
                           Td& dE,
                           const rocblas_stride stE,
                           Id& dinfo,
                           const rocblas_int bc,
                           Wh& hA,
                           Th& hD,
                           Th& hE,
                           Ih& hinfo,
                           double* gpu_time_used,
                           double* cpu_time_used,
                           const rocblas_int hot_calls,
                           const bool perf)
{
    rocblas_int lwork = 5 * n;
    std::vector<T> A;

    if(!perf)
    {
        syev_heev_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        // cpu-lapack performance (only if not in perf mode)
        *cpu_time_used = get_time_us_no_sync();
        for(rocblas_int b = 0; b < bc; ++b)
            cblas_syev_heev<T>(evect, uplo, n, hA[b], lda, hD[b], hE[b], lwork, hinfo[b]);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    syev_heev_initData<true, false, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        syev_heev_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        CHECK_ROCBLAS_ERROR(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA.data(), lda, stA,
                                                dD.data(), stD, dE.data(), stE, dinfo.data(), bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        syev_heev_initData<false, true, T>(handle, evect, n, dA, lda, bc, hA, A, 0);

        start = get_time_us_sync(stream);
        rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA.data(), lda, stA, dD.data(), stD,
                            dE.data(), stE, dinfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <bool BATCHED, bool STRIDED, typename T>
void testing_syev_heev(Arguments argus)
{
    using S = decltype(std::real(T{}));

    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.N;
    rocblas_int lda = argus.lda;
    rocblas_stride stA = argus.bsa;
    rocblas_stride stD = argus.bsb;
    rocblas_stride stE = argus.bsc;
    rocblas_int bc = argus.batch_count;

    char evectC = argus.evect;
    char uploC = argus.uplo_option;
    rocblas_evect evect = char2rocblas_evect(evectC);
    rocblas_fill uplo = char2rocblas_fill(uploC);
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    if(uplo == rocblas_fill_full)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n,
                                                      (T* const*)nullptr, lda, stA, (S*)nullptr, stD,
                                                      (S*)nullptr, stE, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, (T*)nullptr,
                                                      lda, stA, (S*)nullptr, stD, (S*)nullptr, stE,
                                                      (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_value);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(2);

        return;
    }

    // determine sizes
    size_t size_Dres = 0;
    size_t size_Ares = 0;
    size_t size_A = size_t(lda) * n;
    size_t size_D = n;
    size_t size_E = size_D;
    if(argus.unit_check || argus.norm_check)
    {
        size_Ares = size_A;
        size_Dres = size_D;
    }

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (n < 0 || lda < n || bc < 0);
    if(invalid_size)
    {
        if(BATCHED)
            EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n,
                                                      (T* const*)nullptr, lda, stA, (S*)nullptr, stD,
                                                      (S*)nullptr, stE, (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);
        else
            EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, (T*)nullptr,
                                                      lda, stA, (S*)nullptr, stD, (S*)nullptr, stE,
                                                      (rocblas_int*)nullptr, bc),
                                  rocblas_status_invalid_size);

        if(argus.timing)
            ROCSOLVER_BENCH_INFORM(1);

        return;
    }

    // memory size query is necessary
    if(!USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        if(BATCHED)
            CHECK_ALLOC_QUERY(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n,
                                                  (T* const*)nullptr, lda, stA, (S*)nullptr, stD,
                                                  (S*)nullptr, stE, (rocblas_int*)nullptr, bc));
        else
            CHECK_ALLOC_QUERY(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, (T*)nullptr, lda,
                                                  stA, (S*)nullptr, stD, (S*)nullptr, stE,
                                                  (rocblas_int*)nullptr, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations (all cases)
    // host
    host_strided_batch_vector<S> hE(5 * n, 1, 5 * n, bc);
    host_strided_batch_vector<S> hD(size_D, 1, stD, bc);
    host_strided_batch_vector<rocblas_int> hinfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hinfoRes(1, 1, 1, bc);
    host_strided_batch_vector<S> hDres(size_Dres, 1, stD, bc);
    // device
    device_strided_batch_vector<S> dE(size_E, 1, stE, bc);
    device_strided_batch_vector<S> dD(size_D, 1, stD, bc);
    device_strided_batch_vector<rocblas_int> dinfo(1, 1, 1, bc);
    if(size_E)
        CHECK_HIP_ERROR(dE.memcheck());
    if(size_D)
        CHECK_HIP_ERROR(dD.memcheck());
    CHECK_HIP_ERROR(dinfo.memcheck());

    if(BATCHED)
    {
        // memory allocations
        host_batch_vector<T> hA(size_A, 1, bc);
        host_batch_vector<T> hAres(size_Ares, 1, bc);
        device_batch_vector<T> dA(size_A, 1, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA.data(),
                                                      lda, stA, dD.data(), stD, dE.data(), stE,
                                                      dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syev_heev_getError<STRIDED, T>(handle, evect, uplo, n, dA, lda, stA, dD, stD, dE, stE,
                                           dinfo, bc, hA, hAres, hD, hDres, hE, hinfo, hinfoRes,
                                           &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syev_heev_getPerfData<STRIDED, T>(handle, evect, uplo, n, dA, lda, stA, dD, stD, dE,
                                              stE, dinfo, bc, hA, hD, hE, hinfo, &gpu_time_used,
                                              &cpu_time_used, hot_calls, argus.perf);
        }
    }

    else
    {
        // memory allocations
        host_strided_batch_vector<T> hA(size_A, 1, stA, bc);
        host_strided_batch_vector<T> hAres(size_Ares, 1, stA, bc);
        device_strided_batch_vector<T> dA(size_A, 1, stA, bc);
        if(size_A)
            CHECK_HIP_ERROR(dA.memcheck());

        // check quick return
        if(n == 0 || bc == 0)
        {
            EXPECT_ROCBLAS_STATUS(rocsolver_syev_heev(STRIDED, handle, evect, uplo, n, dA.data(),
                                                      lda, stA, dD.data(), stD, dE.data(), stE,
                                                      dinfo.data(), bc),
                                  rocblas_status_success);
            if(argus.timing)
                ROCSOLVER_BENCH_INFORM(0);

            return;
        }

        // check computations
        if(argus.unit_check || argus.norm_check)
        {
            syev_heev_getError<STRIDED, T>(handle, evect, uplo, n, dA, lda, stA, dD, stD, dE, stE,
                                           dinfo, bc, hA, hAres, hD, hDres, hE, hinfo, hinfoRes,
                                           &max_error);
        }

        // collect performance data
        if(argus.timing)
        {
            syev_heev_getPerfData<STRIDED, T>(handle, evect, uplo, n, dA, lda, stA, dD, stD, dE,
                                              stE, dinfo, bc, hA, hD, hE, hinfo, &gpu_time_used,
                                              &cpu_time_used, hot_calls, argus.perf);
        }
    }

    // validate results for rocsolver-test
    // using n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_cout << "\n============================================\n";
            rocsolver_cout << "Arguments:\n";
            rocsolver_cout << "============================================\n";
            if(BATCHED)
            {
                rocsolver_bench_output("evect", "uplo", "n", "lda", "strideD", "strideE", "batch_c");
                rocsolver_bench_output(evect, uplo, n, lda, stD, stE, bc);
            }
            else if(STRIDED)
            {
                rocsolver_bench_output("evect", "uplo", "n", "lda", "strideA", "strideD", "strideE",
                                       "batch_c");
                rocsolver_bench_output(evect, uplo, n, lda, stA, stD, stE, bc);
            }
            else
            {
                rocsolver_bench_output("evect", "uplo", "n", "lda");
                rocsolver_bench_output(evect, uplo, n, lda);
            }
            rocsolver_cout << "\n============================================\n";
            rocsolver_cout << "Results:\n";
            rocsolver_cout << "============================================\n";
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time", "gpu_time", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time", "gpu_time");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_cout << std::endl;
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }
}