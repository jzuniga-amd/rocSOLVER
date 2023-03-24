
/*! \file */
/* ************************************************************************
 * Copyright (c) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#ifndef ROCSPARSE_CHECK_H
#define ROCSPARSE_CHECK_H

#include <exception>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#include "rocsolver_datatype2string.hpp"
#include "rocsparse/rocsparse.h"

#ifndef ROCSPARSE_CHECK
#define ROCSPARSE_CHECK(fcn, error_code)                                                        \
    {                                                                                           \
        rocsparse_status const istat = (fcn);                                                   \
        if(istat != rocsparse_status_success)                                                   \
        {                                                                                       \
            printf("rocsparse API failed at line %d in file %s with error: %s(%d)\n", __LINE__, \
                   __FILE__, rocsparse2string_status(istat), istat);                            \
            fflush(stdout);                                                                     \
            return ((error_code));                                                              \
        };                                                                                      \
    };
#endif

#ifndef THROW_IF_ROCSPARSE_ERROR
#define THROW_IF_ROCSPARSE_ERROR(fcn)                                                    \
    {                                                                                    \
        rocsparse_status const istat = (fcn);                                            \
        if(istat != rocsparse_status_success)                                            \
        {                                                                                \
            printf("rocsparse failed at %s:%d, with error %s(%d)\n", __FILE__, __LINE__, \
                   rocsparse2string_status(istat), istat);                               \
            fflush(stdout);                                                              \
            throw std::runtime_error(__FILE__);                                          \
        };                                                                               \
    };

#endif

#define IDEBUG 0
#define TRACE()                                     \
    {                                               \
        if(IDEBUG >= 1)                             \
        {                                           \
            printf("%s(%d)\n", __FILE__, __LINE__); \
            fflush(stdout);                         \
        };                                          \
    }

#endif
