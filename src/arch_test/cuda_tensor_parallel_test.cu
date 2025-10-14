
#include <cstdint>
#include <ctime>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <cuda_fp16.h>
#include <vector>
// 并行问题，最好所有资源只能够cuda core和tensor core的两个算子同时访问
// 看cuda core占多少寄存器，cuda core占多少，分别看看先
// 只需要2个warp就行了，一个cuda core,一个tensor core，这样会省很多初始值寄存器
// 再看看放到两block看看，能并行不
// 最好看看block能并行不

// 就用4个warp每个block,最少占128个
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" \
                 : "=r"(RD0), "=r"(RD1)                                                                                \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
#define REGS 56
__device__ __forceinline__ void cuda_core_presure(int tid,const half* A, const half* B, half* C,double* cuda_time,half alpha){
    half tmp[REGS * 3];
    #pragma unroll
    for(int i = 0; i < REGS; i++){
        tmp[i] = A[tid * REGS +i];
        tmp[i + REGS] = B[tid * REGS +i];
    }
    clock_t start = clock();
    #pragma unroll
    for(int i = 0; i < 10000; i++){
        if(i % 32 == 0 )
            tmp [0] = tmp[55];
        tmp[2* REGS + (i % REGS)] = __float2half(__half2float(tmp[i % REGS]) * __half2float(tmp[(i % REGS) + REGS]) )+ alpha;
    }
    // __syncthreads();
    clock_t end = clock();
    cuda_time[tid] = (double)(end - start) / CLOCKS_PER_SEC;
    for(int i= 0; i < REGS * 3; i++)
        C[tid] +=tmp[i];
}
// wmma::mma_sync(C_frag[i][j_s], A_frag[reg_load_idx][i], B_frag[reg_load_idx][j_s], C_frag[i][j_s]);
__device__ __forceinline__ void tensor_core_presure(int tid,const half* A, const half* B,half* C, double* tensor_time){ 
    uint32_t RA[8];
    // #pragma unroll
    reinterpret_cast<float4*>(RA)[0] = (reinterpret_cast<const float4 *>(A))[tid];
    reinterpret_cast<float4*>(RA)[1] = (reinterpret_cast<const float4 *>(A))[tid];
    clock_t start = clock();
    #pragma unroll
    for(int i = 0; i < 20000; i++){
        if(i % 32 == 0 ){
            RA[0] = RA[7];
        }
        HMMA16816(RA[6], RA[7], RA[0], RA[1], RA[2], RA[3], RA[4], RA[5], RA[6], RA[7]);
    }
    __syncthreads();
    clock_t end = clock();
    tensor_time[tid ] = (double)(end - start) / CLOCKS_PER_SEC;
    (reinterpret_cast<uint32_t *>(C))[tid] = RA[0];
}


__global__ void core_parallel_test(const half* A, const half* B,half* C,double* time,double* cuda_time,double* tensor_time,const int M, const int N, const int K,half alpha){
    int tid = threadIdx.x  + blockDim.x * blockIdx.x;
    int warp_id = threadIdx.x / 32;
    clock_t start = clock();
    if(warp_id <4){
        cuda_core_presure(tid, A, B,C, cuda_time,alpha);
    }
    else{
    // if(warp_id < 2)
        tensor_core_presure(tid, A, B, C,tensor_time);
        }
    __syncthreads();
    clock_t end = clock();
    time[tid] = (double)(end - start) / CLOCKS_PER_SEC;
}
int main(){
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    half* A = new half[M*K];
    half* B = new half[K*N];
    half* C = new half[M*N];
    half* d_a,*d_b,*d_c;
    double* cuda_time ,*tensor_time,*h_time_cuda, * h_time_tensor,*time,*h_time;
    h_time_cuda = new double[46 * 128 * 2];
    h_time_tensor = new double[46 * 128 * 2];
    h_time = new double[46 * 128 * 2];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for(int i = 0; i < M*K; i++){
        A[i] = __float2half(dis(gen));
    }
    for(int i = 0; i < K*N; i++){
        B[i] = __float2half(dis(gen));
    }

    cudaMalloc(&d_a,M*K*sizeof(half));
    cudaMalloc(&d_b,K*N*sizeof(half));

    cudaMalloc(&cuda_time, 2 * 46 * 128  * sizeof(double));
    cudaMalloc(&tensor_time, 2 * 46 * 128  * sizeof(double));
    cudaMalloc(&time, 2 * 46 * 128  * sizeof(double));
    cudaMalloc(&d_c, 2 * 46 * 128  * sizeof(double));
    cudaMemcpy(d_a, A, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    core_parallel_test<<<46,128 * 2>>>(d_a, d_b,d_c,time,cuda_time,tensor_time ,M, N, K,__float2half(0.1f));
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time elapsed: " << (milliseconds)/1000 << " seconds" << std::endl;
    // 修正内存拷贝
    cudaMemcpy(h_time_cuda, cuda_time, 2 * 46*128*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_time_tensor, tensor_time, 2 * 46*128*sizeof(double), cudaMemcpyDeviceToHost); // 修正这里
    cudaMemcpy(C, d_c, 2 * 46 * 128 *sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_time, time, 2 * 46 * 128 *sizeof(double), cudaMemcpyDeviceToHost);
    // 修正vector初始化
    std::vector<double> cuda_time_vec{h_time_cuda, h_time_cuda + 46*128 * 2};
    std::vector<double> tensor_time_vec(h_time_tensor,h_time_tensor + 46*128 *2);
    std::vector<double> time_vec(h_time,h_time + 46*128 *2);

    // 计算平均值
    double cuda_time_avg = std::accumulate(cuda_time_vec.begin(), cuda_time_vec.end(), 0.0) / cuda_time_vec.size() * 2 ;
    double tensor_time_avg = std::accumulate(tensor_time_vec.begin(), tensor_time_vec.end(), 0.0) / tensor_time_vec.size() * 2;
    double time_avg = std::accumulate(time_vec.begin(), time_vec.end(), 0.0) / time_vec.size();

    std::cout << "CUDA Core average time: " << cuda_time_avg << " seconds" << std::endl;
    std::cout << "Tensor Core average time: " << tensor_time_avg << " seconds" << std::endl;
    std::cout << "Total average time: " << time_avg << " seconds" << std::endl;
    std::cout << "C[0] = "<< __half2float(C[0]) <<"C[46 * 128 +1] = " << __half2float(C[46 * 128 +1]) <<std::endl;
}