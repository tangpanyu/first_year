#include "common.cuh"
#include "mma_ptx.cuh"

#define FETCH_DATA(dst, src, index) reinterpret_cast<float4*>(&dst)[0] = (reinterpret_cast<const float4* >(src))[index]
#define STORE_DATA(dst, src, index) (reinterpret_cast<float4*>(dst))[index] = reinterpret_cast<float4*>(&src)[0]
__global__ void mma_gemm_base(const bf16* A, const bf16* B, bf16* C,const int M, const int N, const int K){
    __shared__ bf16 sA[1024];
    reinterpret_cast<float4*>(&sA)[0] = (reinterpret_cast<const float4* >(A))[0];

}
void run_mma_gemm_base(int M, int N,int K, bf16* A, bf16* B, bf16* C){
    
}