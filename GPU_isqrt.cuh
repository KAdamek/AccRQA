#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef RQA_ISQRT
#define RQA_ISQRT


/*
  Copyright (c) 2021, Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without 
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright 
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

__device__ __inline__ unsigned long long int umul_wide (unsigned int a, unsigned int b)
{
    unsigned long long int r;
    asm ("mul.wide.u32 %0,%1,%2;\n\t" : "=l"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __inline__ uint32_t isqrtll (uint64_t a)
{
    uint64_t rem, arg;
    uint32_t b, r, s, scal;

    arg = a;
    /* Normalize argument */
    scal = __clzll (a) & ~1;
    a = a << scal;
    b = a >> 32;
    /* Approximate rsqrt accurately. Make sure it's an underestimate! */
    float fb, fr;
    fb = (float)b;
    asm ("rsqrt.approx.ftz.f32 %0,%1; \n\t" : "=f"(fr) : "f"(fb));
    r = (uint32_t) fmaf (1.407374884e14f, fr, -438.0f);
    /* Compute sqrt(a) as a * rsqrt(a) */
    s = __umulhi (r, b);
    /* NR iteration combined with back multiply */
    s = s * 2;
    rem = a - umul_wide (s, s);
    r = __umulhi ((uint32_t)(rem >> 32) + 1, r);
    s = s + r;
    /* Denormalize result */
    s = s >> (scal >> 1);
    /* Make sure we get the floor correct; can be off by one to either side */
    rem = arg - umul_wide (s, s);
    if ((int64_t)rem < 0) s--;
    else if (rem >= ((uint64_t)s * 2 + 1)) s++;
    return (arg == 0) ? 0 : s;
}

#endif