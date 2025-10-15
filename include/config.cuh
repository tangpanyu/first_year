#pragma once
#define HOST_DEVICE __host__ __device__ __forceinline__
#define DEVICE __device__ __forceinline__
#define bf16 __nv_bfloat16

#define BM 16
#define BN 8
#define BK 16