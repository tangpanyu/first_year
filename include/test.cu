#include "common.cuh"

#include <cstdint>
#include <cuda_fp16.h>

constexpr int SShift = 3;
constexpr int BBits = 1;
constexpr int MBase=4; 
__global__ void test_swizzle(half* A){
    int tid = threadIdx.x;
    __shared__ half S[64 * 8];
    

    // 先算出在需要swizzle的原矩阵中的偏移映射到smem中，也就是说<3,2，4>每个矩阵都只映射到共享内存的前4列，后面的4列不用管，因为是填充的。
    uint32_t base = (tid >> BBits << 7) + (((tid & ((1 << BBits) -1))) << MBase); // 先算出原行，再算出
    // 加的偏移和shift和bit的表示范围有关，和tid无关。
    uint32_t offset = Swizzle<SShift,BBits,MBase>::apply(static_cast<uint32_t>(base)) + (tid >> (SShift + BBits) << (SShift + BBits + MBase)) ;
    uint32_t ptr = __cvta_generic_to_shared(S + tid * 8);
    printf("trheadIdx.x : %d ,smem addr: %p, offset: %d  \n",tid,__cvta_generic_to_shared(&S[tid*8]),offset / 16);
}

int main(){
    half* ha = new half[8 * 64] ,*d_a;
    for(int i = 0; i < 8 * 64; i++){
        ha[i] = __float2half(i);
    }
    cudaMalloc(&d_a, 8 * 64 * sizeof(half));
    cudaMemcpy(d_a, ha, 8 * 64 * sizeof(half), cudaMemcpyHostToDevice);
    test_swizzle<<<1, 128>>>(d_a);
    cudaDeviceSynchronize();
}