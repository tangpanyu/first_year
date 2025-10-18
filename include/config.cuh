#pragma once
#define HOST_DEVICE __host__ __device__ __forceinline__
#define DEVICE __device__ __forceinline__
#define bf16 __nv_bfloat16

#define BM 16 // mma shape
#define BN 8
#define BK 16

#define STAGE_K 32 // 一个stage加载的K,其中stageM和stageN和BM,BN相同，一个stage加载的是A16*32 ，B 32* 8

#define BLOCK_M 32 // one time stage
#define BLOCK_N 32
#define BLOCK_K 32

#define GRID_M 128
#define GRID_N 128