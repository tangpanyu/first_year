#include "common.cuh"
#include "config.cuh"
#include "mma_ptx.cuh"

#include <cstdint>

// 每个warp负责加载一个16*16的矩阵，也是计算一个16*8*16 * 2的矩阵。
// block的形状是 32 * 32的
// 考虑的MNK都是4096的形状
// block的循环方式，BM是32，就从左到右再到左，

__global__ void mma_gemm_base(const bf16* A, const bf16* B, bf16* C,const int M, const int N, const int K){
    __shared__ bf16 sAB[];

    uint32_t tx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t ty = threadIdx.y + blockIdx.y * blockDim.y;

    uint32_t warp_id = threadIdx.x << 5;
    uint32_t lane_id = threadIdx.x & 31;
    const bf16* block_base_ga, *block_base_gb, *block_base_gc;

    block_base_ga = A + blockIdx.y * BM * K;

    if( blockIdx.y & 1){
        block_base_gb = B + (blockDim.x - 1 - blockIdx.x) * BN * K;
        block_base_gc = C + blockIdx.y * BM * N +  (blockDim.x - 1 -blockIdx.x)* BN;
    }
    else{
        block_base_gb = B + blockIdx.x* BN * K;
        block_base_gc = C + blockIdx.y * BM * N +  blockIdx.x* BN;
    }

    bf16 RA[4];
    bf16 RB[2][2];
    bf16 RC[2][2];
    #pragma unroll
    for(int i = 0; i < 2; ++i)
        #pragma unroll
        for(int j = 0; j < 2; ++j)
            RC[i][j] = 0;

    const bf16* warp_base_ga = block_base_ga + (warp_id >> 1) * MM * K + (warp_id & 1) * MK;
    const bf16* warp_base_gb = block_base_gb + (warp_id >> 1) * MN * K + (warp_id & 1) * MK;
#pragma unroll
    for(int iter =0 ;iter < K /BK; ++iter){

        warp_base_ga += (iter * BK);
        warp_base_gb += (iter * BK);

        bf16* thread_base_ga = (bf16*)warp_base_ga + (lane_id & 15) * K + (lane_id >> 4) * 8 ;
        bf16* thread_base_gb = (bf16*)warp_base_gb + (lane_id & 15) * K + (lane_id >> 4) * 8 ; // 8 * 8 的矩阵 8个线程加载，得改一下。

        uint32_t thread_base_sa = __cvta_generic_to_shared(sAB + threadIdx.x * 8);// address after swizzle;
        uint32_t thread_base_sb = __cvta_generic_to_shared(sAB + BM * BK +  threadIdx.x * 8); 

        cp_async_cg(thread_base_sa, thread_base_ga ,16);
        cp_async_cg(thread_base_sb, thread_base_gb ,16);

        cp_async_commit();
        cp_async_wait_group(0);
        __syncthreads();

        ldmatrix_x4(RA[0], RA[1], RA[2], RA[3], __cvta_generic_to_shared(sAB + lane_id));
        ldmatrix_x4(RB[0][0], RB[0][1], RB[1][0], RB[1][1], __cvta_generic_to_shared(sAB + BM * BK + lane_id));

        hmma16816(RC[0][0], RC[0][1], RA[0], RA[1], RA[2], RA[3], RB[0][0], RB[0][1], RC[0][0], RC[0][1]);
        hmma16816(RC[1][0], RC[1][1], RA[0], RA[1], RA[2], RA[3], RB[1][0], RB[1][1], RC[1][0], RC[1][1]);

        ldmatrix_x4(RA[0], RA[1], RA[2], RA[3], thread_base_sa + MM * MK *16);
        ldmatrix_x4(RB[0][0], RB[0][1], RB[1][0], RB[1][1], thread_base_sb + MN * MK *16);

        hmma16816(RC[0][0], RC[0][1], RA[0], RA[1], RA[2], RA[3], RB[0][0], RB[0][1], RC[0][0], RC[0][1]);
        hmma16816(RC[1][0], RC[1][1], RA[0], RA[1], RA[2], RA[3], RB[1][0], RB[1][1], RC[1][0], RC[1][1]);

    }



    

    
}
void run_mma_gemm_base(int M, int N,int K, bf16* A, bf16* B, bf16* C){
    
}