#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <sys/types.h>
#include <iostream>


static __device__  __forceinline__ void init_mbarrier(uint64_t* bar, uint32_t tx_count){
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));

    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1; \n" 
    :: "r"(bar_ptr), "r"(tx_count));
}

static __device__ __forceinline__ void arrive(uint64_t* bar,uint32_t count=1){
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
    :
    : "r"(bar_ptr), "r"(count)
    : "memory");
}

static __device__ __forceinline__ void expect_byte(uint64_t* bar, uint32_t bytes){
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));

    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
    :: "r"(bar_ptr), "r"(bytes));
}

static __device__ __forceinline__ void wait(uint64_t* bar, uint32_t kPhaseBit){
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));

    asm volatile(
        "{\n"
        ".reg .pred        P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1,[%0], %1;\n"
        "@P1             bra.uni DONE;\n"
        "bra.uni         LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(bar_ptr), "r"(kPhaseBit)
);
}

static __device__ __forceinline__ void asyn_load(half* dst, void const *const src_tma_map, uint64_t* bar, int global_row_idx, int global_col_idx){

    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint64_t tma_ptr = reinterpret_cast<uint64_t>(src_tma_map);
    
    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
    "[%0], [%1, {%3, %4, 0, 0, 0}], [%2];"
    :
    : "r"(dst_ptr),"l"(tma_ptr),"r"(bar_ptr), "r"(global_row_idx), "r"(global_col_idx)
    : "memory");

}
template <int BM, int BK>
__global__ void tma_kernel(const __grid_constant__ CUtensorMap tma_map, half* C, int M, int K){

    __shared__ __align__(128) half smem[BM][BK];
    __shared__ __align__(8) uint64_t full,empty;

    if(threadIdx.x == 0){
        init_mbarrier(&full, 1);
        init_mbarrier(&empty, 1);
    }
    __syncthreads();

    
    if(threadIdx.x == 0){ 
        expect_byte(&full, sizeof(half)*BK*BM);
        wait(&empty, 0);
        asyn_load((half*)smem, &tma_map, &empty, 0, 0);
    }
    if(threadIdx.x == 0)
        arrive(&empty);
    if(threadIdx.x == 0)
        wait(&full,0);

    if(threadIdx.x == 0){
        for(int i=0;i<BM;i++){
            if (i > 0)
                printf("\n");
            for(int j=0;j<BK;j++){
                printf("%f\t",__half2float(smem[i][j]));
            }
        }
    }
}
// The first dimension of the global shape is not divisible, 
// and stride only records tensorRank-1, that is, the divisible dimension is ignored.
template<int BM, int BK,int Swizzle>
__host__ inline static CUtensorMap create_tensor_map(half* A, int M, int K){
    CUtensorMap map;
    void* gmem_address = (void*) A;
    uint64_t globalDim[5] = {(uint64_t)K,(uint64_t)M,1,1,1};
    uint64_t globalStride[5] = { uint64_t(sizeof(half)), uint64_t(sizeof(half) * K),0,0,0};

    uint32_t boxDim[5] = {(uint32_t)BM,(uint32_t)BK,1,1,1};
    uint32_t boxStride[5] = {1 , 1, 1, 1, 1};

    CUresult ret = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 
        5, gmem_address, globalDim, globalStride, boxDim, boxStride,
         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, 
         CU_TENSOR_MAP_L2_PROMOTION_NONE , CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    assert(ret == CUDA_SUCCESS);
    return map;
}

CUtensorMap tma_map;
int main(){
    const int M=64;
    const int K=64;
    const int BM=64;
    const int BK=64;
    const int Swizzle=3;
    half* h_a = new half[M*K];
    half* d_a, *d_c;

    for(int i=0;i<M*K;i++){
        h_a[i] = (i / 8) % 8;
    }
    
    cudaMalloc(&d_a,sizeof(half) * M * K);
    cudaMalloc(&d_c,sizeof(half) * M * K);
    tma_map = create_tensor_map<BM, BK, Swizzle>(d_a, M, K);
    cudaMemcpy(d_a, h_a, sizeof(half) * M * K, cudaMemcpyHostToDevice);
    create_tensor_map<M,K,Swizzle>(d_a,M,K);
    tma_kernel<BM,BK><<<1,1>>>(tma_map,d_c,M,K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_c, sizeof(half) * K * M, cudaMemcpyDeviceToHost);
}