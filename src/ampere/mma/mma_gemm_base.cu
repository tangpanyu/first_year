#include "common.cuh"
#include "config.cuh"
#include "mma_ptx.cuh"

#include <cstdint>

#define FETCH_DATA(dst, src, index) reinterpret_cast<float4*>(&dst)[0] = (reinterpret_cast<const float4* >(src))[index]
#define STORE_DATA(dst, src, index) (reinterpret_cast<float4*>(dst))[index] = reinterpret_cast<float4*>(&src)[0]
__global__ void mma_gemm_base(const bf16* A, const bf16* B, bf16* C,const int M, const int N, const int K){
    __shared__ bf16 sAB[];

    uint32_t tx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t ty = threadIdx.y + blockIdx.y * blockDim.y;

    uint32_t warp_id = threadIdx.x << 5;
    uint32_t lane_id = threadIdx.x & 31;

    const bf16* block_base_ga = A + blockIdx.y * BLOCK_M;
    const bf16* block_base_gb = B + blockIdx.x * BLOCK_N;

    const bf16* warp_base_ga = block_base_ga + (warp_id >> 1) * MM * K + (warp_id & 1) * MK ;
    const bf16* warp_base_gb = block_base_gb + (warp_id >> 1) * MN * K + (warp_id & 1) * MK ;

    bf16* thread_base_ga = (bf16*)warp_base_ga + (lane_id & 16) * K + (lane_id >> 4) * 8 ;
    bf16* thread_base_gb = (bf16*)warp_base_gb + (lane_id & 16) * K + (lane_id >> 4) * 8 ;

    bf16 RA[4];
    bf16 RB[2][2];
    bf16 RC[2][2];
    #pragma unroll
    for(int i = 0; i < 2; ++i)
        #pragma unroll
        for(int j = 0; j < 2; ++j)
            RC[i][j] = 0;


    uint32_t sA = __cvta_generic_to_shared(sAB);
    uint32_t sB = sA + MM * BLOCK_K * sizeof(bf16);

    // 取数据到smem

    cp_async_cg(sA + threadIdx.x * 16, thread_base_ga ,16);
    cp_async_cg(sB + threadIdx.x * 16, thread_base_gb ,16);

    cp_async_commit();
    cp_async_wait_group(0);

    __syncthreads();

    

    
}
void run_mma_gemm_base(int M, int N,int K, bf16* A, bf16* B, bf16* C){
    
}