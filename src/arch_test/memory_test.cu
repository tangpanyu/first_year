#include <iostream>

__global__ void row_read(float* A, float* B){

    __shared__ float sA[4][32];
    float* base_b = B + blockIdx.x;
    float* base_a = A + blockIdx.x * 32 * 4;
    int index = (threadIdx.x << 4) + (threadIdx.x & 15);
    reinterpret_cast<float4*>(sA)[threadIdx.x] = reinterpret_cast<float4*>(base_a)[index];
    __syncthreads();
    float C = 0;
    if(threadIdx.x ==0) {
        for(int i=0;i<32;++i)
            C = sA[0][i] + sA[1][i] + sA[2][i] + sA[3][i] + C;
    }
    base_b[0] = C;
}

__global__ void col_read(float* A,float* B){
    __shared__ float sA[4][32];
    float* base_a = A + blockIdx.x * 32 * 4;
    float* base_b = B + blockIdx.x;
    int index = threadIdx.x;
    reinterpret_cast<float4*>(sA)[threadIdx.x] = reinterpret_cast<float4*>(base_a)[index];
    __syncthreads();
        float C = 0;
    if(threadIdx.x ==0) {
        for(int i=0;i<32;++i)
            C = sA[0][i] + sA[1][i] + sA[2][i] + sA[3][i] + C;
    }
    base_b[0] = C;
}

int main(){
    float* A,*B;
    cudaMallocManaged(&B, 8192*8192 * sizeof(float));
    cudaMallocManaged(&A, 8192*8192 * sizeof(float));
    for(int i=0;i<8192*8192;++i)
        A[i] = i % 16;

            col_read<<<256, 32>>>(A,B);
    cudaDeviceSynchronize();
    printf("done,B[32] = %f \n",B[32]);
    row_read<<<256, 32>>>(A,B);

    cudaDeviceSynchronize();
    printf("done,B[32] = %f \n",B[32]);

    

}