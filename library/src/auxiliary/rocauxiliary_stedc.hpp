/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#define STEDC_BDIM 512 // Number of threads per thread-block used in main stedc kernels
#define MAXITERS 50 // Max number of iterations for root finding method

typedef enum rocsolver_stedc_mode_
{
    rocsolver_stedc_mode_qr,
    rocsolver_stedc_mode_jacobi,
    rocsolver_stedc_mode_bisection
} rocsolver_stedc_mode;

template <rocsolver_stedc_mode MODE>
__host__ __device__ inline rocblas_int stedc_num_levels(const rocblas_int n);

/***************** Device auxiliary functions ******************************************/
/**************************************************************************************/

//--------------------------------------------------------------------------------------//
/** SEQ_EVAL evaluates the secular equation at a given point. It accumulates the
    corrections to the elements in D so that distance to poles are computed
   accurately **/
template <typename S>
__device__ void seq_eval(const rocblas_int type,
                         const rocblas_int k,
                         const rocblas_int dd,
                         S* D,
                         const S* z,
                         const S p,
                         const S cor,
                         S* pt_fx,
                         S* pt_fdx,
                         S* pt_gx,
                         S* pt_gdx,
                         S* pt_hx,
                         S* pt_hdx,
                         S* pt_er,
                         bool modif)
{
    S er, fx, gx, hx, fdx, gdx, hdx, zz, tmp;
    rocblas_int gout, hout;

    // prepare computations
    // if type = 0: evaluate secular equation
    if(type == 0)
    {
        gout = k + 1;
        hout = k;
    }
    // if type = 1: evaluate secular equation without the k-th pole
    else if(type == 1)
    {
        if(modif)
        {
            tmp = D[k] - cor;
            D[k] = tmp;
        }
        gout = k;
        hout = k;
    }
    // if type = 2: evaluate secular equation without the k-th and (k+1)-th poles
    else if(type == 2)
    {
        if(modif)
        {
            tmp = D[k] - cor;
            D[k] = tmp;
            tmp = D[k + 1] - cor;
            D[k + 1] = tmp;
        }
        gout = k;
        hout = k + 1;
    }
    else
    {
        // unexpected value for type, something is wrong
        assert(false);
    }

    // computations
    gx = 0;
    gdx = 0;
    er = 0;
    for(int i = 0; i < gout; ++i)
    {
        tmp = D[i] - cor;
        if(modif)
            D[i] = tmp;
        zz = z[i];
        tmp = zz / tmp;
        gx += zz * tmp;
        gdx += tmp * tmp;
        er += gx;
    }
    er = abs(er);

    hx = 0;
    hdx = 0;
    for(int i = dd - 1; i > hout; --i)
    {
        tmp = D[i] - cor;
        if(modif)
            D[i] = tmp;
        zz = z[i];
        tmp = zz / tmp;
        hx += zz * tmp;
        hdx += tmp * tmp;
        er += hx;
    }

    fx = p + gx + hx;
    fdx = gdx + hdx;

    // return results
    *pt_fx = fx;
    *pt_fdx = fdx;
    *pt_gx = gx;
    *pt_gdx = gdx;
    *pt_hx = hx;
    *pt_hdx = hdx;
    *pt_er = er;
}

//--------------------------------------------------------------------------------------//
/** SEQ_SOLVE solves secular equation at point k (i.e. computes kth eigenvalue
   that is within an internal interval). We use rational interpolation and fixed
   weights method between the 2 poles of the interval. (TODO: In the future, we
   could consider using 3 poles for those cases that may need it to reduce the
   number of required iterations to converge. The performance improvements are
   expected to be marginal, though) **/
template <typename S>
__device__ rocblas_int seq_solve(const rocblas_int dd,
                                 S* D,
                                 const S* z,
                                 const S p,
                                 rocblas_int k,
                                 S* ev,
                                 const S tol,
                                 const S ssfmin,
                                 const S ssfmax)
{
    bool converged = false;
    bool up, fixed;
    S lowb, uppb, aa, bb, cc, x;
    S nx, er, fx, fdx, gx, gdx, hx, hdx, oldfx;
    S tau, eta;
    S dk, dk1, ddk, ddk1;
    rocblas_int kk;
    rocblas_int k1 = k + 1;

    // initialize
    dk = D[k];
    dk1 = D[k1];
    x = (dk + dk1) / 2; // midpoint of interval
    tau = (dk1 - dk);
    S pinv = 1 / p;

    // find bounds and initial guess; translate origin
    seq_eval(2, k, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx, &er, false);
    gdx = z[k] * z[k];
    hdx = z[k1] * z[k1];
    fx = cc + 2 * (hdx - gdx) / tau;
    if(fx > 0)
    {
        // if the secular eq at the midpoint is positive, the root is in between
        // D[k] and the midpoint take D[k] as the origin, i.e. x = D[k] + tau with
        // tau in (0, uppb)
        lowb = 0;
        uppb = tau / 2;
        up = true;
        kk = k; // origin remains the same
        aa = cc * tau + gdx + hdx;
        bb = gdx * tau;
        eta = sqrt(abs(aa * aa - 4 * bb * cc));
        if(aa > 0)
            tau = 2 * bb / (aa + eta);
        else
            tau = (aa - eta) / (2 * cc);
        x = dk + tau; // initial guess
    }
    else
    {
        // otherwise, the root is in between the midpoint and D[k+1]
        // take D[k+1] as the origin, i.e. x = D[k+1] + tau with tau in (lowb, 0)
        lowb = -tau / 2;
        uppb = 0;
        up = false;
        kk = k + 1; // translate the origin
        aa = cc * tau - gdx - hdx;
        bb = hdx * tau;
        eta = sqrt(abs(aa * aa + 4 * bb * cc));
        if(aa < 0)
            tau = 2 * bb / (aa - eta);
        else
            tau = -(aa + eta) / (2 * cc);
        x = dk1 + tau; // initial guess
    }

    // evaluate secular eq and get input values to calculate step correction
    seq_eval(0, kk, dd, D, z, pinv, (up ? dk : dk1), &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
    seq_eval(1, kk, dd, D, z, pinv, tau, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
    bb = z[kk];
    aa = bb / D[kk];
    fdx += aa * aa;
    bb *= aa;
    fx += bb;

    // calculate tolerance er for convergence test
    er += 8 * (hx - gx) + 2 * pinv + 3 * abs(bb) + abs(tau) * fdx;

    // if the value of secular eq is small enough, no point to continue;
    // converged!!!
    if(abs(fx) <= tol * er)
        converged = true;

    // otherwise...
    else
    {
        // update bounds
        lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
        uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

        // calculate first step correction with fixed weight method
        ddk = D[k];
        ddk1 = D[k1];
        if(up)
            cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
        else
            cc = fx - ddk * fdx - (dk1 - dk) * z[k1] * z[k1] / ddk1 / ddk1;
        aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
        bb = ddk * ddk1 * fx;
        if(cc == 0)
        {
            if(aa == 0)
            {
                if(up)
                    aa = z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx);
                else
                    aa = z[k1] * z[k1] + ddk * ddk * (gdx + hdx);
            }
            eta = bb / aa;
        }
        else
        {
            eta = sqrt(abs(aa * aa - 4 * bb * cc));
            if(aa <= 0)
                eta = (aa - eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa + eta);
        }

        // verify that the correction eta will get x closer to the root
        // i.e. eta*fx should be negative. If not the case, take a Newton step
        // instead
        if(fx * eta >= 0)
            eta = -fx / fdx;

        // now verify that applying the correction won't get the process out of
        // bounds if that is the case, bisect the interval instead
        if(tau + eta > uppb || tau + eta < lowb)
        {
            if(fx < 0)
                eta = (uppb - tau) / 2;
            else
                eta = (lowb - tau) / 2;
        }

        // take the step
        tau += eta;
        x = (up ? dk : dk1) + tau;

        // evaluate secular eq and get input values to calculate step correction
        oldfx = fx;
        seq_eval(1, kk, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
        bb = z[kk];
        aa = bb / D[kk];
        fdx += aa * aa;
        bb *= aa;
        fx += bb;

        // calculate tolerance er for convergence test
        er += 8 * (hx - gx) + 2 * pinv + 3 * abs(bb) + abs(tau) * fdx;

        // from now on, further step corrections will be calculated either with
        // fixed weights method or with normal interpolation depending on the value
        // of boolean fixed
        cc = up ? -1 : 1;
        fixed = (cc * fx) > (abs(oldfx) / 10);

        // MAIN ITERATION LOOP
        // ==============================================
        for(int i = 1; i < MAXITERS; ++i)
        {
            // if the value of secular eq is small enough, no point to continue;
            // converged!!!
            if(abs(fx) <= tol * er)
            {
                converged = true;
                break;
            }

            // update bounds
            lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
            uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

            // calculate next step correction with either fixed weight method or
            // simple interpolation
            ddk = D[k];
            ddk1 = D[k1];
            if(fixed)
            {
                if(up)
                    cc = fx - ddk1 * fdx - (dk - dk1) * z[k] * z[k] / ddk / ddk;
                else
                    cc = fx - ddk * fdx - (dk1 - dk) * z[k1] * z[k1] / ddk1 / ddk1;
            }
            else
            {
                if(up)
                    gdx += aa * aa;
                else
                    hdx += aa * aa;
                cc = fx - ddk * gdx - ddk1 * hdx;
            }
            aa = (ddk + ddk1) * fx - ddk * ddk1 * fdx;
            bb = ddk * ddk1 * fx;
            if(cc == 0)
            {
                if(aa == 0)
                {
                    if(fixed)
                    {
                        if(up)
                            aa = z[k] * z[k] + ddk1 * ddk1 * (gdx + hdx);
                        else
                            aa = z[k1] * z[k1] + ddk * ddk * (gdx + hdx);
                    }
                    else
                        aa = ddk * ddk * gdx + ddk1 * ddk1 * hdx;
                }
                eta = bb / aa;
            }
            else
            {
                eta = sqrt(abs(aa * aa - 4 * bb * cc));
                if(aa <= 0)
                    eta = (aa - eta) / (2 * cc);
                else
                    eta = (2 * bb) / (aa + eta);
            }

            // verify that the correction eta will get x closer to the root
            // i.e. eta*fx should be negative. If not the case, take a Newton step
            // instead
            if(fx * eta >= 0)
                eta = -fx / fdx;

            // now verify that applying the correction won't get the process out of
            // bounds if that is the case, bisect the interval instead
            if(tau + eta > uppb || tau + eta < lowb)
            {
                if(fx < 0)
                    eta = (uppb - tau) / 2;
                else
                    eta = (lowb - tau) / 2;
            }

            // take the step
            tau += eta;
            x = (up ? dk : dk1) + tau;

            // evaluate secular eq and get input values to calculate step correction
            oldfx = fx;
            seq_eval(1, kk, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
            bb = z[kk];
            aa = bb / D[kk];
            fdx += aa * aa;
            bb *= aa;
            fx += bb;

            // calculate tolerance er for convergence test
            er += 8 * (hx - gx) + 2 * pinv + 3 * abs(bb) + abs(tau) * fdx;

            // update boolean fixed if necessary
            if(fx * oldfx > 0 && abs(fx) > abs(oldfx) / 10)
                fixed = !fixed;
        }
    }

    *ev = x;
    return converged ? 0 : 1;
}

//--------------------------------------------------------------------------------------//
/** SEQ_SOLVE_EXT solves secular equation at point n (i.e. computes last
   eigenvalue). We use rational interpolation and fixed weights method between
   the (n-1)th and nth poles. (TODO: In the future, we could consider using 3
   poles for those cases that may need it to reduce the number of required
   iterations to converge. The performance improvements are expected to be
   marginal, though) **/
template <typename S>
__device__ rocblas_int seq_solve_ext(const rocblas_int dd,
                                     S* D,
                                     const S* z,
                                     const S p,
                                     S* ev,
                                     const S tol,
                                     const S ssfmin,
                                     const S ssfmax)
{
    bool converged = false;
    S lowb, uppb, aa, bb, cc, x;
    S er, fx, fdx, gx, gdx, hx, hdx;
    S tau, eta;
    S dk, dkm1, ddk, ddkm1;
    rocblas_int k = dd - 1;
    rocblas_int km1 = dd - 2;

    // initialize
    dk = D[k];
    dkm1 = D[km1];
    x = dk + p / 2;
    S pinv = 1 / p;

    // find bounds and initial guess
    seq_eval(2, km1, dd, D, z, pinv, x, &cc, &fdx, &gx, &gdx, &hx, &hdx, &er, false);
    gdx = z[km1] * z[km1];
    hdx = z[k] * z[k];
    fx = cc + gdx / (dkm1 - x) - 2 * hdx * pinv;
    if(fx > 0)
    {
        // if the secular eq at the midpoint is positive, the root is in between
        // D[k] and the midpoint take D[k] as the origin, i.e. x = D[k] + tau with
        // tau in (0, uppb)
        lowb = 0;
        uppb = p / 2;
        tau = dk - dkm1;
        aa = -cc * tau + gdx + hdx;
        bb = hdx * tau;
        eta = sqrt(aa * aa + 4 * bb * cc);
        if(aa < 0)
            tau = 2 * bb / (eta - aa);
        else
            tau = (aa + eta) / (2 * cc);
    }
    else
    {
        // otherwise, the root is in between the midpoint and D[k+1]
        // take D[k+1] as the origin, i.e. x = D[k+1] + tau with tau in (lowb, 0)
        lowb = p / 2;
        uppb = p;
        eta = gdx / (dk - dkm1 + p) + hdx / p;
        if(cc <= eta)
            tau = p;
        else
        {
            tau = dk - dkm1;
            aa = -cc * tau + gdx + hdx;
            bb = hdx * tau;
            eta = sqrt(aa * aa + 4 * bb * cc);
            if(aa < 0)
                tau = 2 * bb / (eta - aa);
            else
                tau = (aa + eta) / (2 * cc);
        }
    }
    x = dk + tau; // initial guess

    // evaluate secular eq and get input values to calculate step correction
    seq_eval(0, km1, dd, D, z, pinv, dk, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);
    seq_eval(0, km1, dd, D, z, pinv, tau, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);

    // calculate tolerance er for convergence test
    er += abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;

    // if the value of secular eq is small enough, no point to continue;
    // converged!!!
    if(abs(fx) <= tol * er)
        converged = true;

    // otherwise...
    else
    {
        // update bounds
        lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
        uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

        // calculate first step correction with fixed weight method
        ddk = D[k];
        ddkm1 = D[km1];
        cc = abs(fx - ddkm1 * gdx - ddk * hdx);
        aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
        bb = ddk * ddkm1 * fx;
        if(cc == 0)
        {
            eta = uppb - tau;
        }
        else
        {
            eta = sqrt(abs(aa * aa - 4 * bb * cc));
            if(aa >= 0)
                eta = (aa + eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa - eta);
        }

        // verify that the correction eta will get x closer to the root
        // i.e. eta*fx should be negative. If not the case, take a Newton step
        // instead
        if(fx * eta > 0)
            eta = -fx / (gdx + hdx);

        // now verify that applying the correction won't get the process out of
        // bounds if that is the case, bisect the interval instead
        if(tau + eta > uppb || tau + eta < lowb)
        {
            if(fx < 0)
                eta = (uppb - tau) / 2;
            else
                eta = (lowb - tau) / 2;
        }

        // take the step
        tau += eta;
        x = dk + tau;

        // evaluate secular eq and get input values to calculate step correction
        seq_eval(0, km1, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);

        // calculate tolerance er for convergence test
        er += abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;

        // MAIN ITERATION LOOP
        // ==============================================
        for(int i = 1; i < MAXITERS; ++i)
        {
            // if the value of secular eq is small enough, no point to continue;
            // converged!!!
            if(abs(fx) <= tol * er)
            {
                converged = true;
                break;
            }

            // update bounds
            lowb = (fx <= 0) ? std::max(lowb, tau) : lowb;
            uppb = (fx > 0) ? std::min(uppb, tau) : uppb;

            // calculate step correction
            ddk = D[k];
            ddkm1 = D[km1];
            cc = fx - ddkm1 * gdx - ddk * hdx;
            aa = (ddk + ddkm1) * fx - ddk * ddkm1 * (gdx + hdx);
            bb = ddk * ddkm1 * fx;
            eta = sqrt(abs(aa * aa - 4 * bb * cc));
            if(aa >= 0)
                eta = (aa + eta) / (2 * cc);
            else
                eta = (2 * bb) / (aa - eta);

            // verify that the correction eta will get x closer to the root
            // i.e. eta*fx should be negative. If not the case, take a Newton step
            // instead
            if(fx * eta > 0)
                eta = -fx / (gdx + hdx);

            // now verify that applying the correction won't get the process out of
            // bounds if that is the case, bisect the interval instead
            if(tau + eta > uppb || tau + eta < lowb)
            {
                if(fx < 0)
                    eta = (uppb - tau) / 2;
                else
                    eta = (lowb - tau) / 2;
            }

            // take the step
            tau += eta;
            x = dk + tau;

            // evaluate secular eq and get input values to calculate step correction
            seq_eval(0, km1, dd, D, z, pinv, eta, &fx, &fdx, &gx, &gdx, &hx, &hdx, &er, true);

            // calculate tolerance er for convergence test
            er += abs(tau) * (hdx + gdx) - 8 * (hx + gx) - hx + pinv;
        }
    }

    *ev = x;
    return converged ? 0 : 1;
}

//--------------------------------------------------------------------------------------//
/** STEDC_NUM_LEVELS returns the ideal number of times/levels in which a matrix
   (or split block) will be divided during the divide phase of divide & conquer
   algorithm. i.e. number of sub-blocks = 2^levels **/
template <>
__host__ __device__ inline rocblas_int stedc_num_levels<rocsolver_stedc_mode_qr>(const rocblas_int n)
{
    rocblas_int levels = 0;
    // return the max number of levels such that the sub-blocks are at least of
    // size 1 (i.e. 2^levels <= n), and there are no more than 256 sub-blocks
    // (i.e. 2^levels <= 256)
    if(n <= 2)
        return levels;

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
/** STEDC_SPLIT finds independent blocks (split-blocks) in the tridiagonal
   matrix given by D and E. The independent blocks can then be solved in
    parallel by the DC algorithm.
        - Call this kernel with batch_count single-threaded groups in x **/
template <typename S>
ROCSOLVER_KERNEL void stedc_split(const rocblas_int n,
                                  S* DD,
                                  const rocblas_stride strideD,
                                  S* EE,
                                  const rocblas_stride strideE,
                                  rocblas_int* splitsA,
                                  const S eps)
{
    rocblas_int bid = hipBlockIdx_x;

    // select batch instance
    S* D = DD + (bid * strideD);
    S* E = EE + (bid * strideE);
    rocblas_int* splits = splitsA + bid * (5 * n + 2);

    rocblas_int k = 0; // position where the last block starts
    S tol; // tolerance. If an element of E is <= tol we have an independent
        // block
    rocblas_int bs; // size of an independent block
    rocblas_int nb = 1; // number of blocks
    splits[0] = 0; // positions where each block begings

    // main loop
    while(k < n)
    {
        bs = 1;
        for(rocblas_int j = k; j < n - 1; ++j)
        {
            tol = eps * sqrt(abs(D[j])) * sqrt(abs(D[j + 1]));
            if(abs(E[j]) < tol)
            {
                // Split next independent block
                // save its location in matrix
                splits[nb] = j + 1;
                nb++;
                break;
            }
            bs++;
        }
        k += bs;
    }
    splits[nb] = n;
    splits[n + 1] = nb; // also save the number of split blocks
}

//--------------------------------------------------------------------------------------//
/** STEDC_DIVIDE_KERNEL implements the divide phase of the DC algorithm. It
   divides each split-block into a number of sub-blocks.
        - Call this kernel with batch_count groups in x. Groups are of size
   STEDC_BDIM.
        - If there are actually more split-blocks than STEDC_BDIM, some threads
   will work with more than one split-block sequentially. **/
template <rocsolver_stedc_mode MODE, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_divide_kernel(const rocblas_int n,
                                                                        S* DD,
                                                                        const rocblas_stride strideD,
                                                                        S* EE,
                                                                        const rocblas_stride strideE,
                                                                        rocblas_int* splitsA)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_x;
    // split-block id
    rocblas_int sid = hipThreadIdx_x;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
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

    // work with STEDC_BDIM split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_BDIM)
    {
        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks in split-block
        levs = stedc_num_levels<MODE>(bs);
        blks = 1 << levs;

        // 1. DIVIDE PHASE
        /* ----------------------------------------------------------------- */
        // (artificially divide split-block into blks sub-blocks
        // find initial positions of each sub-blocks)

        // find sizes of sub-blocks
        ns[0] = bs;
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

        // find beginning of sub-blocks and update D elements
        p2 = p1;
        ps[0] = p2;
        for(int i = 1; i < blks; ++i)
        {
            p2 += ns[i - 1];
            ps[i] = p2;

            // perform sub-block division
            p = E[p2 - 1];
            D[p2] -= p;
            D[p2 - 1] -= p;
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_SOLVE_KERNEL implements the solver phase of the DC algorithm to
        compute the eigenvalues/eigenvectors of the different sub-blocks of each
   split-block. A matrix in the batch could have many split-blocks, and each
   split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS
   groups in y. Groups are single-thread.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
   be analysed in parallel). If there are actually more split-blocks, some
   groups will work with more than one split-block sequentially. **/
template <typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM) stedc_solve_kernel(const rocblas_int n,
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
    S* W = WA + bid * (2 * n);
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
        levs = stedc_num_levels<rocsolver_stedc_mode_qr>(bs);
        blks = 1 << levs;

        // 2. SOLVE PHASE
        /* ----------------------------------------------------------------- */
        // Solve the blks sub-blocks in parallel.

        if(tid < blks)
        {
            sbs = ns[tid];
            p2 = ps[tid];

            // (Until STEQR is parallelized, only the first thread associated
            // with each sub-block does computations)
            if(tidb == 0)
                run_steqr(sbs, D + p2, E + p2, C + p2 + p2 * ldc, ldc, info, W + p2 * 2, 30 * bs,
                          eps, ssfmin, ssfmax, false);
            __syncthreads();
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEPREPARE_KERNEL performs deflation and prepares the secular equation for
    every pair of sub-blocks that need to be merged in a split block. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, and STEDC_NUM_SPLIT_BLKS groups in y.
          Groups are size nn.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a thread has an id larger than the actual number of sub-blocks
          in a split-block, it will do nothing. **/
template <rocsolver_stedc_mode MODE, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergePrepare_kernel(const rocblas_int k,
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
                              const S eps,
                              const S ssfmin,
                              const S ssfmax)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split block id
    rocblas_int sid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int tid = hipThreadIdx_x;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = psA + n;
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
    /* --------------------------------------------------- */

    // temporary arrays in shared memory
    /* --------------------------------------------------- */
    extern __shared__ rocblas_int lsmem[];
    // used to store temp values during the different reductions
    S* inrms = reinterpret_cast<S*>(lsmem);
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
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
        __syncthreads();

        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        levs = stedc_num_levels<MODE>(bs);
        blks = 1 << levs;

        // 3. MERGE PHASE
        /* ----------------------------------------------------------------- */
        // Work with merges on level k. One thread per sub-block is necessary
        if(tid < blks && k < levs)
        {
            rocblas_int iam, sz, bdm, dim;
            S* ptz;
            rocblas_int bd = 1 << k;
            bdm = bd << 1;

            // tid indexes the sub-blocks in the entire split block
            // iam indexes the sub-blocks in the context of the merge
            // (according to its level in the merge tree)
            iam = tid % bdm;
            p2 = ps[tid];

            // 3a. find rank-1 modification components (z and p) for this merge
            /* ----------------------------------------------------------------- */
            // Threads with iam < bd work with components above the merge point;
            //  threads with iam >= bd work below the merge point
            if(iam < bd)
            {
                sz = ns[tid];
                for(int j = 1; j < bd - iam; ++j)
                    sz += ns[tid + j];
                // with this, all threads involved in a merge (above merge point)
                // will point to the same row of C and the same off-diag element
                ptz = C + p2 - 1 + sz;
                p = 2 * E[p2 - 1 + sz];
            }
            else if(iam >= bd)
            {
                sz = 0;
                for(int j = 0; j < iam - bd; ++j)
                    sz += ns[tid - j - 1];
                // with this, all threads involved in a merge (below merge point)
                // will point to the same row of C and the same off-diag element
                ptz = C + p2 - sz;
                p = 2 * E[p2 - sz - 1];
            }

            // copy elements of z
            for(int j = 0; j < ns[tid]; ++j)
                z[p2 + j] = ptz[(p2 + j) * ldc] / sqrt(2);
            /* ----------------------------------------------------------------- */

            // 3b. calculate deflation tolerance
            /* ----------------------------------------------------------------- */
            S valf, valg, maxd, maxz;

            // first compute maximum of diagonal and z in each thread block
            maxd = std::abs(D[p2]);
            maxz = std::abs(z[p2]);
            for(int i = 1; i < ns[tid]; ++i)
            {
                valf = std::abs(D[p2 + i]);
                valg = std::abs(z[p2 + i]);
                maxd = valf > maxd ? valf : maxd;
                maxz = valg > maxz ? valg : maxz;
            }
            inrms[tid] = maxd;
            inrms[tid + blks] = maxz;
            __syncthreads();

            // now follow reduction process
            if(iam == 0)
            {
                for(int i = 1; i < bdm; ++i)
                {
                    valf = inrms[tid + i];
                    valg = inrms[tid + blks + i];
                    maxd = valf > maxd ? valf : maxd;
                    maxz = valg > maxz ? valg : maxz;
                }
                inrms[tid] = maxd;
                inrms[tid + blks] = maxz;
            }
            __syncthreads();

            // tol should be  8 * eps * (max diagonal or z element participating in
            // merge)
            maxd = inrms[tid - iam];
            maxz = inrms[tid - iam + blks];
            maxd = maxz > maxd ? maxz : maxd;

            S tol = 8 * eps * maxd;
            /* ----------------------------------------------------------------- */

            // 3c. deflate eigenvalues
            /* ----------------------------------------------------------------- */
            S f, g, c, s, r;

            // first deflate each sub-block
            for(int i = 0; i < ns[tid]; ++i)
            {
                g = z[p2 + i];
                if(abs(p * g) <= tol)
                {
                    // deflated ev because component in z is zero
                    idd[p2 + i] = 0;
                }
                else
                {
                    rocblas_int jj = 1;
                    valg = D[p2 + i];
                    for(int j = 0; j < i; ++j)
                    {
                        if(idd[p2 + j] == 1 && abs(D[p2 + j] - valg) <= tol)
                        {
                            // deflated ev because it is repeated
                            idd[p2 + i] = 0;
                            // rotation to eliminate component in z
                            f = z[p2 + j];
                            lartg(f, g, c, s, r);
                            z[p2 + j] = r;
                            z[p2 + i] = 0;
                            // update C with the rotation
                            for(int ii = 0; ii < n; ++ii)
                            {
                                valf = C[ii + (p2 + j) * ldc];
                                valg = C[ii + (p2 + i) * ldc];
                                C[ii + (p2 + j) * ldc] = valf * c - valg * s;
                                C[ii + (p2 + i) * ldc] = valf * s + valg * c;
                            }
                            break;
                        }
                        jj++;
                    }
                    if(jj > i)
                    {
                        // non-deflated ev
                        idd[p2 + i] = 1;
                    }
                }
            }

            // then compare with other sub-blocks participating in this merge
            // following a simple, reduction-like process.
            for(int ii = 0; ii <= k; ++ii)
            {
                rocblas_int div = 1 << (ii + 1);
                //actual number of threads is halved each time
                if(iam % div == div - 1)
                {
                    // find limits
                    rocblas_int inb = (1 << ii) - 1;
                    rocblas_int inc = div - 1;
                    rocblas_int countb = ns[tid];
                    rocblas_int countc = 0;
                    for(int i = inc; i > inb; --i)
                        countc += ns[tid - i];
                    for(int i = inb; i > 0; --i)
                        countb += ns[tid - i];
                    inb = ps[tid - inb];
                    inc = ps[tid - inc];

                    // perform comparisons
                    for(int i = 0; i < countb; ++i)
                    {
                        if(idd[inb + i] == 1)
                        {
                            valg = D[inb + i];
                            for(int j = 0; j < countc; ++j)
                            {
                                if(idd[inc + j] == 1 && abs(D[inc + j] - valg) <= tol)
                                {
                                    // deflated ev because it is repeated
                                    idd[inb + i] = 0;
                                    // rotation to eliminate component in z
                                    g = z[inb + i];
                                    f = z[inc + j];
                                    lartg(f, g, c, s, r);
                                    z[inc + j] = r;
                                    z[inb + i] = 0;
                                    // update C with the rotation
                                    for(int ii = 0; ii < n; ++ii)
                                    {
                                        valf = C[ii + (inc + j) * ldc];
                                        valg = C[ii + (inb + i) * ldc];
                                        C[ii + (inc + j) * ldc] = valf * c - valg * s;
                                        C[ii + (inb + i) * ldc] = valf * s + valg * c;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
                __syncthreads();
            }
            /* ----------------------------------------------------------------- */

            // 3d.1. Organize data with non-deflated values to prepare secular equation
            /* ----------------------------------------------------------------- */
            // determine boundaries of what would be the new merged sub-block
            // 'in' will be its initial position
            rocblas_int in = ps[tid - iam];
            // 'sz' will be its size (i.e. the sum of the sizes of all merging
            // sub-blocks)
            sz = ns[tid];
            for(int i = iam; i > 0; --i)
                sz += ns[tid - i];
            for(int i = bdm - 1 - iam; i > 0; --i)
                sz += ns[tid + i];

            // define shifted arrays
            S* tmpd = temps + in * n;
            S* diag = D + in;
            rocblas_int* mask = idd + in;
            S* zz = z + in;
            rocblas_int* per = pers + in;

            // find degree and components of secular equation
            // tmpd contains the non-deflated diagonal elements (ie. poles of the
            // secular eqn) zz contains the corresponding non-zero elements of the
            // rank-1 modif vector
            rocblas_int dd = 0;
            for(int i = 0; i < sz; ++i)
            {
                if(mask[i] == 1)
                {
                    if(iam == 0)
                    {
                        per[dd] = i;
                        tmpd[dd] = p < 0 ? -diag[i] : diag[i];
                        if(dd != i)
                            zz[dd] = zz[i];
                    }
                    dd++;
                }
            }
            /* ----------------------------------------------------------------- */
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVALUES_KERNEL solves the secular equation for
    every pair of sub-blocks that need to be merged in a split block. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as half of the unmerged sub-blocks in current level. Each group works
          with a merge. Groups are size STEDC_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than half the actual number of unmerged sub-blocks
          in the level, it will do nothing. **/
template <rocsolver_stedc_mode MODE, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeValues_kernel(const rocblas_int k,
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
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split block id
    rocblas_int sid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int mid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int tid;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = psA + n;
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
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    rocblas_int tn;
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        __syncthreads();

        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        // tn is the number of thread-groups needed
        levs = stedc_num_levels<MODE>(bs);
        blks = levs - 1 - k;
        tn = (blks < 0) ? 0 : 1 << blks;
        blks = 1 << levs;

        // 3. MERGE PHASE
        /* ----------------------------------------------------------------- */
        // Work with merges on level k. A thread-group works with two leaves in the merge tree;
        // all threads work together to solve the secular equation.
        if(mid < tn)
        {
            rocblas_int iam, sz, bdm, dim;
            S* ptz;
            S valf, valg;
            rocblas_int bd = 1 << k;
            bdm = bd << 1;
            dim = hipBlockDim_x / bdm;

            // iam indexes the sub-blocks in the context of the merge
            // (according to its level in the merge tree)
            iam = tidb / dim;
            // tid indexes the sub-blocks in the entire split block
            tid = mid * bdm + iam;
            p2 = ps[tid];

            // Find off-diagonal element of the merge
            if(iam < bd && tid < blks)
            {
                sz = ns[tid];
                for(int j = 1; j < bd - iam; ++j)
                    sz += ns[tid + j];
                p = 2 * E[p2 - 1 + sz];
            }
            else if(iam >= bd && tid < blks)
            {
                sz = 0;
                for(int j = 0; j < iam - bd; ++j)
                    sz += ns[tid - j - 1];
                p = 2 * E[p2 - sz - 1];
            }

            // 3d.2. Organize data with non-deflated values to prepare secular equation
            /* ----------------------------------------------------------------- */
            // determine boundaries of what would be the new merged sub-block
            // 'in' will be its initial position
            rocblas_int in = ps[tid - iam];
            // 'sz' will be its size (i.e. the sum of the sizes of all merging sub-blocks)
            sz = ns[tid];
            for(int i = iam; i > 0; --i)
                sz += ns[tid - i];
            for(int i = bdm - 1 - iam; i > 0; --i)
                sz += ns[tid + i];

            // final number of merged sub-blocks
            rocblas_int tsz = 1 << (levs - 1 - k);
            tsz = (bs - 1) / tsz + 1;

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
            for(int i = 0; i < tsz; ++i)
            {
                if(i < dd)
                {
                    if(i % 2 == 0)
                    {
                        for(int j = iam; j < dd / 2; j += bdm)
                        {
                            if(tmpd[2 * j] > tmpd[2 * j + 1])
                            {
                                valf = tmpd[2 * j];
                                tmpd[2 * j] = tmpd[2 * j + 1];
                                tmpd[2 * j + 1] = valf;
                                valf = zz[2 * j];
                                zz[2 * j] = zz[2 * j + 1];
                                zz[2 * j + 1] = valf;
                                bd = per[2 * j];
                                per[2 * j] = per[2 * j + 1];
                                per[2 * j + 1] = bd;
                            }
                        }
                    }
                    else
                    {
                        for(int j = iam; j < (dd - 1) / 2; j += bdm)
                        {
                            if(tmpd[2 * j + 1] > tmpd[2 * j + 2])
                            {
                                valf = tmpd[2 * j + 1];
                                tmpd[2 * j + 1] = tmpd[2 * j + 2];
                                tmpd[2 * j + 2] = valf;
                                valf = zz[2 * j + 1];
                                zz[2 * j + 1] = zz[2 * j + 2];
                                zz[2 * j + 2] = valf;
                                bd = per[2 * j + 1];
                                per[2 * j + 1] = per[2 * j + 2];
                                per[2 * j + 2] = bd;
                            }
                        }
                    }
                }
                __syncthreads();
            }

            // make dd copies of the non-deflated ordered diagonal elements
            // (i.e. the poles of the secular eqn) so that the distances to the
            // eigenvalues (D - lambda_i) are updated while computing each eigenvalue.
            // This will prevent collapses and division by zero when an eigenvalue
            // is too close to a pole.
            for(int j = iam + 1; j < sz; j += bdm)
            {
                for(int i = 0; i < dd; ++i)
                    tmpd[i + j * n] = tmpd[i];
            }

            // finally copy over all diagonal elements in ev. ev will be overwritten
            // by the new computed eigenvalues of the merged block
            for(int i = iam; i < sz; i += bdm)
                ev[i] = diag[i];
            __syncthreads();
            /* ----------------------------------------------------------------- */

            // 3e. Solve secular eqns, i.e. find the dd zeros
            // corresponding to non-deflated new eigenvalues of the merged block
            /* ----------------------------------------------------------------- */
            // each thread will find a different zero in parallel
            S a, b;
            for(int j = iam; j < sz; j += bdm)
            {
                if(mask[j] == 1)
                {
                    // find position in the ordered array
                    int cc = 0;
                    valf = p < 0 ? -ev[j] : ev[j];
                    for(int jj = 0; jj < dd; ++jj)
                    {
                        if(tmpd[jj + j * n] == valf)
                            break;
                        else
                            cc++;
                    }

                    // computed zero will overwrite 'ev' at the corresponding position.
                    // 'tmpd' will be updated with the distances D - lambda_i.
                    // deflated values are not changed.
                    rocblas_int linfo;
                    if(cc == dd - 1)
                        linfo = seq_solve_ext(dd, tmpd + j * n, zz, (p < 0 ? -p : p), ev + j, eps,
                                              ssfmin, ssfmax);
                    else
                        linfo = seq_solve(dd, tmpd + j * n, zz, (p < 0 ? -p : p), cc, ev + j, eps,
                                          ssfmin, ssfmax);
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
                        if(p > 0)
                            valf *= (per[i] == j) ? valg : valg / (diag[per[i]] - diag[j]);
                        else
                            valf *= (per[i] == j) ? valg : -valg / (diag[per[i]] - diag[j]);
                    }
                }
                valf = sqrt(-valf);
                zz[i] = zz[i] < 0 ? -valf : valf;
            }
            /* ----------------------------------------------------------------- */
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVECTORS_KERNEL merges vectors from the secular equation for
    every pair of sub-blocks that need to be merged in a split block. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as columns would be in the matrix if its size is exact multiple of nn.
          Each group works with a column. Groups are size STEDC_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than the actual number of columns n,
          it will do nothing. **/
template <rocsolver_stedc_mode MODE, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeVectors_kernel(const rocblas_int k,
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
                              const S eps,
                              const S ssfmin,
                              const S ssfmax)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split block id
    rocblas_int sid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int mid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid, vidb;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    S* E = EE + bid * strideE;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = psA + n;
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
    /* --------------------------------------------------- */

    // temporary arrays in shared memory
    /* --------------------------------------------------- */
    extern __shared__ rocblas_int lsmem[];
    // used to store temp values during the different reductions
    S* inrms = reinterpret_cast<S*>(lsmem);
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    rocblas_int tn;
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        __syncthreads();

        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        levs = stedc_num_levels<MODE>(bs);
        blks = 1 << levs;

        // tn is max number of vectors in each sub-block
        tn = (bs - 1) / blks + 1;

        // 3. MERGE PHASE
        /* ----------------------------------------------------------------- */
        // Work with merges on level k. Each thread-group works with one vector.
        if(mid < tn * blks && k < levs)
        {
            rocblas_int iam, sz, bdm;
            S* ptz;
            S valf, valg;
            rocblas_int bd = 1 << k;
            bdm = bd << 1;

            // tid indexes the sub-blocks in the entire split block
            tid = mid / tn;
            p2 = ps[tid];
            // vidb indexes the vectors associated with each sub-block
            vidb = mid % tn;
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

            // final number of merged sub-blocks
            rocblas_int tsz = 1 << (levs - 1 - k);
            tsz = (bs - 1) / tsz + 1;

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

            // 3f. Compute vectors corresponding to non-deflated values
            /* ----------------------------------------------------------------- */
            S temp, nrm;
            rocblas_int j = vidb;
            bool go = (j < ns[tid] && idd[p2 + j] == 1);

            if(go)
            {
                // compute vectors of rank-1 perturbed system and their norms
                nrm = 0;
                for(int i = tidb; i < dd; i += dim)
                {
                    valf = zz[i] / temps[i + (p2 + j) * n];
                    nrm += valf * valf;
                    temps[i + (p2 + j) * n] = valf;
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
                nrm = sqrt(nrm);

                // multiply by C (row by row)
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
            /* ----------------------------------------------------------------- */
        }
    }
}

//--------------------------------------------------------------------------------------//
/** STEDC_MERGEVECTORS_KERNEL updates vectors after merges are done. A matrix in the batch
    could have many split-blocks, and each split-block could be divided in a maximum of nn sub-blocks.
        - Call this kernel with batch_count groups in z, STEDC_NUM_SPLIT_BLKS groups in y,
          and as many groups as columns would be in the matrix if its size is exact multiple of nn.
          Each group works with a column. Groups are size STEDC_BDIM.
        - STEDC_NUM_SPLIT_BLKS is fixed (is the number of split-blocks that will
          be analysed in parallel). If there are actually more split-blocks, some
          groups will work with more than one split-block sequentially.
        - An upper bound for the number of sub-blocks (nn) can be estimated from
          the size n. If a group has an id larger than the actual number of columns n,
          it will do nothing. **/
template <rocsolver_stedc_mode MODE, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(STEDC_BDIM)
    stedc_mergeUpdate_kernel(const rocblas_int k,
                             const rocblas_int n,
                             S* DD,
                             const rocblas_stride strideD,
                             S* CC,
                             const rocblas_int shiftC,
                             const rocblas_int ldc,
                             const rocblas_stride strideC,
                             S* tmpzA,
                             S* vecsA,
                             rocblas_int* splitsA,
                             const S eps,
                             const S ssfmin,
                             const S ssfmax)
{
    // threads and groups indices
    /* --------------------------------------------------- */
    // batch instance id
    rocblas_int bid = hipBlockIdx_z;
    // split block id
    rocblas_int sid = hipBlockIdx_y;
    // merge sub-block id
    rocblas_int mid = hipBlockIdx_x;
    // thread id
    rocblas_int tidb = hipThreadIdx_x;
    rocblas_int dim = hipBlockDim_x;
    rocblas_int tid, vidb;
    /* --------------------------------------------------- */

    // select batch instance to work with
    /* --------------------------------------------------- */
    S* C;
    if(CC)
        C = load_ptr_batch<S>(CC, bid, shiftC, strideC);
    S* D = DD + bid * strideD;
    /* --------------------------------------------------- */

    // temporary arrays in global memory
    /* --------------------------------------------------- */
    // contains the beginning of split blocks
    rocblas_int* splits = splitsA + bid * (5 * n + 2);
    // the sub-blocks sizes
    rocblas_int* nsA = splits + n + 2;
    // the sub-blocks initial positions
    rocblas_int* psA = nsA + n;
    // if idd[i] = 0, the value in position i has been deflated
    rocblas_int* idd = psA + n;
    // the rank-1 modification vectors in the merges
    S* z = tmpzA + bid * (2 * n);
    // roots of secular equations
    S* evs = z + n;
    // updated eigenvectors after merges
    S* vecs = vecsA + bid * 2 * (n * n);
    /* --------------------------------------------------- */

    // temporary arrays in shared memory
    /* --------------------------------------------------- */
    extern __shared__ rocblas_int lsmem[];
    // used to store temp values during the different reductions
    S* inrms = reinterpret_cast<S*>(lsmem);
    /* --------------------------------------------------- */

    // local variables
    /* --------------------------------------------------- */
    // total number of split blocks
    rocblas_int nb = splits[n + 1];
    // size of split block
    rocblas_int bs;
    // beginning of split block
    rocblas_int p1;
    // beginning of sub-block
    rocblas_int p2;
    // number of sub-blocks
    rocblas_int blks;
    // number of level of division
    rocblas_int levs;
    // other aux variables
    rocblas_int tn;
    S p;
    rocblas_int *ns, *ps;
    /* --------------------------------------------------- */

    // work with STEDC_NUM_SPLIT_BLKS split blocks in parallel
    /* --------------------------------------------------- */
    for(int kb = sid; kb < nb; kb += STEDC_NUM_SPLIT_BLKS)
    {
        __syncthreads();

        // Select current split block
        p1 = splits[kb];
        p2 = splits[kb + 1];
        bs = p2 - p1;
        ns = nsA + p1;
        ps = psA + p1;

        // determine ideal number of sub-blocks
        levs = stedc_num_levels<MODE>(bs);
        blks = 1 << levs;

        // tn is max number of vectors in each sub-block
        tn = (bs - 1) / blks + 1;

        // 3. MERGE PHASE
        /* ----------------------------------------------------------------- */
        // Work with merges on level k. Each thread-group works with one vector.
        if(mid < tn * blks && k < levs)
        {
            rocblas_int iam, sz, bdm;
            S* ptz;
            S valf, valg;
            rocblas_int bd = 1 << k;
            bdm = bd << 1;

            // tid indexes the sub-blocks in the entire split block
            tid = mid / tn;
            p2 = ps[tid];
            // vidb indexes the vectors associated with each sub-block
            vidb = mid % tn;
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

            // 3g. update D and C with computed values and vectors
            /* ----------------------------------------------------------------- */
            rocblas_int j = vidb;
            bool go = (j < ns[tid] && idd[p2 + j] == 1);
            if(go)
            {
                if(tidb == 0)
                    D[p2 + j] = evs[p2 + j];
                for(int i = in + tidb; i < in + sz; i += dim)
                    C[i + (p2 + j) * ldc] = vecs[i + (p2 + j) * n];
            }
            /* ----------------------------------------------------------------- */
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
/** This local gemm adapts rocblas_gemm to multiply complex*real, and
    overwrite result: A = A*B **/
template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename S,
          typename U,
          std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void local_gemm(rocblas_handle handle,
                const rocblas_int n,
                U A,
                const rocblas_int shiftA,
                const rocblas_int lda,
                const rocblas_stride strideA,
                S* B,
                S* temp,
                S* work,
                const rocblas_int shiftT,
                const rocblas_int ldt,
                const rocblas_stride strideT,
                const rocblas_int batch_count,
                S** workArr)
{
    // Execute A*B -> temp -> A

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    S one = 1.0;
    S zero = 0.0;

    // temp = A*B
    rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, A,
                     shiftA, lda, strideA, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt,
                     strideT, batch_count, workArr);

    // A = temp
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL(copy_mat<T>, dim3(blocks, blocks, batch_count), dim3(BS2, BS2), 0,
                            stream, copymat_from_buffer, n, n, A, shiftA, lda, strideA, temp);

    rocblas_set_pointer_mode(handle, old_mode);
}

template <bool BATCHED,
          bool STRIDED,
          typename T,
          typename S,
          typename U,
          std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void local_gemm(rocblas_handle handle,
                const rocblas_int n,
                U A,
                const rocblas_int shiftA,
                const rocblas_int lda,
                const rocblas_stride strideA,
                S* B,
                S* temp,
                S* work,
                const rocblas_int shiftT,
                const rocblas_int ldt,
                const rocblas_stride strideT,
                const rocblas_int batch_count,
                S** workArr)
{
    // Execute A -> work; work*B -> temp -> A

    // everything must be executed with scalars on the host
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
    S one = 1.0;
    S zero = 0.0;

    // work = real(A)
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_int blocks = (n - 1) / BS2 + 1;
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_to_buffer, n, n, A, shiftA, lda,
                            strideA, work, rocblas_fill_full);

    // temp = work*B
    rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work,
                     shiftT, ldt, strideT, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt,
                     strideT, batch_count, workArr);

    // real(A) = temp
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, true>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, A, shiftA, lda,
                            strideA, temp, rocblas_fill_full);

    // work = imag(A)
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_to_buffer, n, n, A, shiftA, lda,
                            strideA, work, rocblas_fill_full);

    // temp = work*B
    rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, n, n, &one, work,
                     shiftT, ldt, strideT, B, shiftT, ldt, strideT, &zero, temp, shiftT, ldt,
                     strideT, batch_count, workArr);

    // imag(A) = temp
    ROCSOLVER_LAUNCH_KERNEL((copy_mat<T, S, false>), dim3(blocks, blocks, batch_count),
                            dim3(BS2, BS2), 0, stream, copymat_from_buffer, n, n, A, shiftA, lda,
                            strideA, temp, rocblas_fill_full);

    rocblas_set_pointer_mode(handle, old_mode);
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
        size_t s1, s2;

        // requirements for solver of small independent blocks
        rocsolver_steqr_getMemorySize<T, S>(evect, n, batch_count, &s1);

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
        // constants
        S eps = get_epsilon<S>();
        S ssfmin = get_safemin<S>();
        S ssfmax = S(1.0) / ssfmin;
        ssfmin = sqrt(ssfmin) / (eps * eps);
        ssfmax = sqrt(ssfmax) / S(3.0);
        rocblas_int blocksn = (n - 1) / BS2 + 1;

        // initialize identity matrix in C if required
        if(evect == rocblas_evect_tridiagonal)
            ROCSOLVER_LAUNCH_KERNEL(init_ident<T>, dim3(blocksn, blocksn, batch_count),
                                    dim3(BS2, BS2), 0, stream, n, n, C, shiftC, ldc, strideC);

        // initialize identity matrix in tempvect
        rocblas_int ldt = n;
        rocblas_stride strideT = n * n;
        ROCSOLVER_LAUNCH_KERNEL(init_ident<S>, dim3(blocksn, blocksn, batch_count), dim3(BS2, BS2),
                                0, stream, n, n, tempvect, 0, ldt, strideT);

        // find max number of sub-blocks to consider during the divide phase
        rocblas_int maxlevs = stedc_num_levels<rocsolver_stedc_mode_qr>(n);
        rocblas_int maxblks = 1 << maxlevs;

        // find independent split blocks in matrix
        ROCSOLVER_LAUNCH_KERNEL(stedc_split, dim3(batch_count), dim3(1), 0, stream, n, D + shiftD,
                                strideD, E + shiftE, strideE, splits_map, eps);

        // 1. divide phase
        //-----------------------------
        ROCSOLVER_LAUNCH_KERNEL((stedc_divide_kernel<rocsolver_stedc_mode_qr, S>),
                                dim3(batch_count), dim3(STEDC_BDIM), 0, stream, n, D + shiftD,
                                strideD, E + shiftE, strideE, splits);

        // 2. solve phase
        //-----------------------------
        ROCSOLVER_LAUNCH_KERNEL((stedc_solve_kernel<S>),
                                dim3(maxblks, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(1), 0,
                                stream, n, D + shiftD, strideD, E + shiftE, strideE, tempvect, 0,
                                ldt, strideT, info, (S*)work_stack, splits, eps, ssfmin, ssfmax);

        // 3. merge phase
        //----------------
        size_t lmemsize1 = sizeof(S) * maxblks;
        size_t lmemsize3 = sizeof(S) * STEDC_BDIM;
        rocblas_int numgrps3 = ((n - 1) / maxblks + 1) * maxblks;

        // launch merge for level k
        /** TODO: using max number of levels for now. Kernels return immediately when passing
            the actual number of levels in the split block. We should explore if synchronizing
            to copy back the actual number of levels makes any difference **/
        for(rocblas_int k = 0; k < maxlevs; ++k)
        {
            // a. prepare secular equations
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergePrepare_kernel<rocsolver_stedc_mode_qr, S>),
                                    dim3(1, STEDC_NUM_SPLIT_BLKS, batch_count), dim3(maxblks),
                                    lmemsize1, stream, k, n, D + shiftD, strideD, E + shiftE,
                                    strideE, tempvect, 0, ldt, strideT, tmpz, tempgemm, splits, eps,
                                    ssfmin, ssfmax);

            // b. solve to find merged eigen values
            rocblas_int numgrps2 = 1 << (maxlevs - 1 - k);
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeValues_kernel<rocsolver_stedc_mode_qr, S>),
                                    dim3(numgrps2, STEDC_NUM_SPLIT_BLKS, batch_count),
                                    dim3(STEDC_BDIM), 0, stream, k, n, D + shiftD, strideD,
                                    E + shiftE, strideE, tmpz, tempgemm, splits, eps, ssfmin, ssfmax);

            // c. find merged eigen vectors
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeVectors_kernel<rocsolver_stedc_mode_qr, S>),
                                    dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count),
                                    dim3(STEDC_BDIM), lmemsize3, stream, k, n, D + shiftD, strideD,
                                    E + shiftE, strideE, tempvect, 0, ldt, strideT, tmpz, tempgemm,
                                    splits, eps, ssfmin, ssfmax);

            // c. update level
            ROCSOLVER_LAUNCH_KERNEL((stedc_mergeUpdate_kernel<rocsolver_stedc_mode_qr, S>),
                                    dim3(numgrps3, STEDC_NUM_SPLIT_BLKS, batch_count),
                                    dim3(STEDC_BDIM), lmemsize3, stream, k, n, D + shiftD, strideD,
                                    tempvect, 0, ldt, strideT, tmpz, tempgemm, splits, eps, ssfmin,
                                    ssfmax);
        }

        // 4. update and sort
        //----------------------
        // eigenvectors C <- C*tempvect
        local_gemm<BATCHED, STRIDED, T>(handle, n, C, shiftC, ldc, strideC, tempvect, tempgemm,
                                        static_cast<S*>(work_stack), 0, ldt, strideT, batch_count,
                                        workArr);

        // finally sort eigenvalues and eigenvectors
        auto const nblocks = batch_count;
        ROCSOLVER_LAUNCH_KERNEL((stedc_sort<T>), dim3(1, 1, nblocks), dim3(BS1), 0, stream, n,
                                D + shiftD, strideD, C, shiftC, ldc, strideC, batch_count,
                                splits_map);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
