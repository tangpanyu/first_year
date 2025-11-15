
#include <common.cuh>
#include <cstdint>
#include <mma_ptx.cuh>
#include <cuda_fp16.h>
#include <assert.h>
#include <memory>
#include <sys/types.h>
#include <cublas_v2.h>
#include <random>
#define checkRuntime(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA runtime error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
/*
我想取64x64的，因为这样硬件也能做两层流水，block launch的开销就能抵消，如果不做就得干等了。
这样的话可以做3层流水，硬件两层流水了。但是ILP会低一点，得试一下。
而且64 x 64 可以小粒度一点，L2的cache会容易命中。
但是SM有4个tensor core,也就是block内没有warp调度。
    目的：
        1.既然只能是persistent kernel，肯定要把一个Kernel的ILP拉到最大，也就是寄存器和smem都占满
        2.感觉吧最好是能两个block的流水，所以尽量把一半占满先，然后切换能做流水，只有一个硬件没法流水了。
        3.减少warp切换，切换也要成本，所以说shape和warp要取平衡。
    
    寄存器异步：
        寄存器 STAGE 异步做不了，寄存器做做stage 3的寄存器超了，固定至少需要120个，所以不做3,只做2.
    
    所以整体的架构就是Block硬件两层流水，Shared Memory 3层流水，寄存器 2层流水。

    问题：
        1.mma是warp指令，SM有4个tensor core,导致没有warp调度，会导致第一个block的数据加载没有被覆盖
        但是第二个的block可以被前面的计算和访存覆盖了。
        2.这和只有一个persistent kernel的区别就是后面的block数据加载延时可以被覆盖。
        3.这可能是ILP和TLP的综合，好像还可以。
        4.Persistent Kernel的意思是Block数少于SM个数，所以得写三种：
            1. 也就是现在写的，至少保证2 STAGES
            2. 进一步增大分块shape,只在Block保证流水
            3. Persistent Kernel，只保证底层流水且通过增加ILP和循环保证Block不切换
    两层硬件流水的好处：
        1.目的是要计算单元一直结算，如果是persistent kernel,这hopper不一样，hopper的数据加载是完全异步的。
            1.hopper的异步是由另外的线程发起的，但是ampere的异步拷贝是由计算线程发起的，所以存在一定顺序：就是在计算前发起下一次。
        2.主要是能覆盖第一个blockd的数据写回
        3.因为ampere的异步拷贝是由计算线程发起的，所以得拷贝回之后才能进入下一个block 的计算逻辑。
        4.既然说block加载到sm中，都变成了warp，所以2个block4个warp和1个block8个warp的区别
            1.在ampere中，persistent kernle写法，在当前C的block计算完成切换到下一个Cblock时，没法发起下一次数据加载和计算
            2.在hopper中，persistent kernel，数据由生产者加载，和计算无关。cooperate就是无法用计算覆盖epilogue,所以有了pingpong.
            3.但是ampere的mbarrier不好用，所以无法用warp specialize,而且不是warp group,不好同步。
            4.所以我的解法是2个block一个SM,覆盖掉每个block的epilogue.
            5.8个warp的数据是依赖的，得一次加载进shared，但是数据是以warp写回的，所以说瓶颈在prologus.
            6.切分成2个block和硬件流水，prologue和epilogue都可以覆盖掉。
    L2cache问题：
        1.对N轴进行分割，每次循环到最下面，然后最下面到最上面，保证B的复用，至少可以保证B是只用加载一次，A的一行也可以复用
        2.比如N是3个block,，可以保证至少75%的命中率。
        3.因为SM限制了block的启动数量，所以多个block暂时认为没用，多个block的数据复用B也是可以保证。

    BBit == 0，swizzle无效，因为主要是在取数据排布上存在问题，在写数据就是顺序写。
*/


#define WM 32
#define WN 32
#define WK 32

#define BM 64
#define BN 64
#define BK 64

#define SShift 3
#define BBits 1
#define MBase 4

#define MM 16 //mmashape
#define MN 8
#define MK 16

#define WARP_SIZE 32

#define RA_I WM/MM * WK/MK
#define RB_I WN/MN * WK/MK
#define RC_I WN/MN * WM/MM

#define STAGES 3 // 只需要保证有Stage - 1 个流水在走就行了，因为有一个要被用到。
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))


// 这每个线程都算一遍，不太好吧，都是一样的。


template<int NUM_SM, int TM, int TN>
struct Schedule{
    int block;
    int it;
    int total_blocks_m, total_blocks_n;
    __device__ __forceinline__ Schedule(int M, int N, int _block) {
        block = _block;
        it = 0;
        total_blocks_m = CEIL_DIV(M, BM);
        total_blocks_n = CEIL_DIV(N, BN);
        assert(CEIL_DIV(M, BM)%TM == 0 && total_blocks_n%TN == 0);
    }

    __device__ __forceinline__ bool next(int &block_m, int& block_n) {
        int num = it*NUM_SM + block;
        if (num >= total_blocks_m*total_blocks_n) {return false;}
        
        int cur_tile = num / (TM*TN);
        int cur_tile_pos = num % (TM*TN); 
        block_m = TM*(cur_tile / (total_blocks_n/TN));
        block_n = TN*(cur_tile % (total_blocks_n/TN));

        block_n += cur_tile_pos % TN;
        if((block_n / TN) & 1){
            // printf("cur_tile_pos = %d \n",cur_tile_pos);
            block_m = block_m + TM - 1 - (cur_tile_pos / TN);
        }
        else{
            block_m += cur_tile_pos / TN;
        }

        if((block_m / TM) & 1){
            block_n = total_blocks_n - 1 - block_n;
        }
        
        ++it;
        return true;
    }
};


struct SMem {
    alignas(128) half A[BM*BK*STAGES];
    alignas(128) half B[BK*BN*STAGES];
};
template<int NUM_THREADS, int NUM_SM>
__global__ void mma_gemm_xstage_f16(const half* const __restrict__ A, const half* const __restrict__ B, half* __restrict__ C,const int M, const int N, const int K){
    
    extern __shared__ __align__(128) uint8_t smem[];
    SMem &s = *reinterpret_cast<SMem*>(smem);
    half *sA = s.A;
    half *sB = s.B;
    uint32_t warp_id = threadIdx.y;
    uint32_t lane_id = threadIdx.x;
    const half* block_base_ga, *block_base_gb;
    half* block_base_gc;

    Schedule<NUM_SM, 8, 8> scheduler(M, N, blockIdx.x);
    int block_m, block_n;
    while(scheduler.next(block_m, block_n)){
    
        block_base_ga = A + block_m * BM * K;
        block_base_gb = B + block_n* BN * K;
        block_base_gc = C + block_m * BM * N +  block_n * BN;


        const half* warp_base_ga = block_base_ga + warp_id * MK; 
        const half* warp_base_gb = block_base_gb + warp_id * MK;

        uint32_t RA[2][RA_I][4],RB[2][RA_I][4],RC[RA_I][4];

        half* thread_base_ga = (half*)warp_base_ga + (lane_id & 15) * K + (lane_id >> 4) * 8;  // 8 和4都是不是通解
        half* thread_base_gb = (half*)warp_base_gb + (lane_id & 15) * K + (lane_id >> 4) * 8; 

        half* thread_base_sa = sA + (warp_id * warpSize + lane_id) * 8; // 线程加载的都是顺序值。
        half* thread_base_sb = sB + (warp_id * warpSize + lane_id) * 8;

        uint32_t s_stage_w = 0,s_stage_r = 0;
    
        #pragma unroll
        for(int k = 0; k < STAGES - 1; ++k,++s_stage_w){
            #pragma unroll
            for(int i = 0; i < 4; i++){ // 4 先写死
                //  BM * BK 是一个block的块， MM*BK是一个加载的块，可以看作是 i* warps * warpsize * 8
                // MM *K是一次加载A的一MM行，MN列。
                cp_async_cg(__cvta_generic_to_shared(thread_base_sa + s_stage_w * BM * BK + i * MM * BK), \
                thread_base_ga + i * MM * K, 16);
                cp_async_cg(__cvta_generic_to_shared(thread_base_sb + s_stage_w * BN * BK + i * MM * BK), \
                thread_base_gb + i * MM * K, 16);
                
            }
            thread_base_ga += BK;
            thread_base_gb += BK;

            cp_async_commit();
        }

        const uint32_t warp_a_idx = warp_id >> 1;
        const uint32_t warp_b_idx = (warp_id & 1) << 1;


        for(int iter = 0; iter < K / BK ; ++iter){
            
            if(iter < K / BK - 2){
                #pragma unroll
                for(int i = 0; i < 4; i++){ 
                    
                    cp_async_cg(__cvta_generic_to_shared(thread_base_sa + s_stage_w * BM * BK + i * MM * BK), \
                    thread_base_ga + i * MM * K, 16);
                    cp_async_cg(__cvta_generic_to_shared(thread_base_sb + s_stage_w * BN * BK + i * MM * BK), \
                    thread_base_gb + i * MM * K, 16);
                }

                thread_base_ga += BK;
                thread_base_gb += BK;
                (++s_stage_w) %= STAGES;
            }
            cp_async_commit();
            cp_async_wait_group(STAGES-1);
            __syncthreads();

            
            // smem中不存在行号要求，以为都是0swizzle的，所以只需要定位块，和定位块内地址就行了，块内地址就是lane_id * 8
            uint32_t offset =  s_stage_r * BM * BK + warp_a_idx * WM * BK + lane_id * 8;

            #pragma unroll
            for(int i = 0; i < 4; i++){
                
                LDMATRIX_X4(RA[0][i][0], RA[0][i][1], RA[0][i][2], RA[0][i][3], \
                    __cvta_generic_to_shared(sA + offset + i * MM * MK)); 

                LDMATRIX_X4(RB[0][i][0], RB[0][i][2], RB[0][i][1], RB[0][i][3], \
                    __cvta_generic_to_shared(sB + offset + i * MN * MK)); 
            }

            #pragma unroll
            for(int i = 0; i < 4; i++){
                
                HMMA16816(  RC[0][0],RC[0][1],
                            RA[0][i][0],RA[0][i][1],RA[0][i][2],RA[0][i][3],
                            RB[0][i][0],RB[0][i][1],
                            RC[0][0],RC[0][1]);

                HMMA16816(  RC[0][2],RC[0][3],
                            RA[0][i][0],RA[0][i][1],RA[0][i][2],RA[0][i][3],
                            RB[0][i][2],RB[0][i][3],
                            RC[0][2],RC[0][3]);
            }

            #pragma unroll
            for(int i = 0; i < 4; i++){
                // smem中不存在行号要求，以为都是0swizzle的，所以只需要定位块，和定位块内地址就行了，块内地址就是lane_id * 8
                
                LDMATRIX_X4(RA[1][i][0], RA[1][i][1], RA[1][i][2], RA[1][i][3], \
                    __cvta_generic_to_shared(sA + offset + i * MM * MK)); 

                LDMATRIX_X4(RB[1][i][0], RB[1][i][2], RB[1][i][1], RB[1][i][3], \
                    __cvta_generic_to_shared(sB + offset + i * MN * MK)); 
            }

            #pragma unroll
            for(int i = 0; i < 4; i++){
                
                HMMA16816(  RC[1][0],RC[1][1],
                            RA[0][i][0],RA[0][i][1],RA[0][i][2],RA[0][i][3],
                            RB[1][i][0],RB[0][1][1],
                            RC[1][0],RC[1][1]);

                HMMA16816(  RC[1][2],RC[1][3],
                            RA[0][i][0],RA[0][i][1],RA[0][i][2],RA[0][i][3],
                            RB[1][i][2],RB[1][i][3],
                            RC[1][2],RC[1][3]);
            }

            #pragma unroll
            for(int i = 0; i < 4; i++){
                
                HMMA16816(  RC[2][0],RC[2][1],
                            RA[1][i][0],RA[1][i][1],RA[1][i][2],RA[1][i][3],
                            RB[0][i][0],RB[0][i][1],
                            RC[2][0],RC[2][1]);

                HMMA16816(  RC[2][2],RC[2][3],
                            RA[1][i][1],RA[0][i][1],RA[1][i][2],RA[1][i][3],
                            RB[0][i][2],RB[0][i][3],
                            RC[2][2],RC[2][3]);
            }

            #pragma unroll
            for(int i = 0; i < 4; i++){
                
                HMMA16816(  RC[3][0],RC[3][1],
                            RA[1][i][0],RA[1][i][1],RA[1][i][2],RA[1][i][3],
                            RB[1][i][0],RB[1][i][1],
                            RC[3][0],RC[3][1]);

                HMMA16816(  RC[2][2],RC[2][3],
                            RA[1][i][1],RA[0][i][1],RA[1][i][2],RA[1][i][3],
                            RB[1][i][2],RB[1][i][3],
                            RC[3][2],RC[3][3]);
            }

            (++s_stage_r) %= STAGES;

        }
        
        for(int i = 0; i < 4; i++){
            uint32_t* sC = (uint32_t*)sA + warp_id * WM * (WM >> 1) + i * MM * (MN >> 1);

            sC[lane_id] = RC[i][0];
            sC[lane_id + 32] = RC[i][1];
            sC[lane_id + 32 * 2] = RC[i][2];
            sC[lane_id + 32 * 3] = RC[i][3];
        }
        __syncthreads();

        // int4* sC = (int4*)sA + 
    }
}

template<typename T>
__global__ void gemm(T *A, T *B, T *D, size_t M, size_t N, size_t K){
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(ty >= M || tx >= N){
        return;
    }
    T sum = 0.f;
    for(size_t i=0; i< K; ++i){
        sum += A[ty * K + i] * B[tx * K + i];
    }
    D[ty * N + tx] = sum;
}
template<typename T>
__global__ void compare_result(T *A, T *B, size_t M, size_t N){
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(ty >= M || tx >= N){
        return;
    }
    if(A[ty * N + tx] != B[ty * N + tx]){
        printf(" Error! %f \n",__half2float(A[ty * N + tx] - B[ty * N + tx]) );
    }
}
void transposeWithBackup(half* matrix, size_t rows, size_t cols) {
    // 申请空间保存原来的矩阵
    half* backupMatrix = new half[rows * cols];

    // 复制原矩阵内容到备份矩阵
    for (size_t i = 0; i < rows * cols; ++i) {
        backupMatrix[i] = matrix[i];
    }

    // 转置矩阵
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            matrix[j * rows + i] = backupMatrix[i * cols + j];
        }
    }

    // 释放备份矩阵的空间
    delete[] backupMatrix;
}

void rowMajorToColumnMajor(const half* rowMajorMatrix, half* columnMajorMatrix, size_t rows, size_t cols) {
    // 将行主序矩阵转换为列主序矩阵
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            columnMajorMatrix[j * rows + i] = rowMajorMatrix[i * cols + j];
        }
    }
}
void printBinary(uint16_t value) {
    for (int i = 15; i >= 0; i--) {
        printf("%d", (value >> i) & 1);
        if (i % 4 == 0) printf(" "); // 每4位添加空格以提高可读性
    }
    printf("\n");
}

__global__ void cuda_hgemm(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,const int M, const int N, const int K){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    half sum = 0.f;
    if (x >= N || y >= M) return;
    for(int i=0;i<K;++i){
        sum += A[x * K + i] * B[y * K + i];
    }
    C[y * N + x] = sum;
}
int main(){
    const size_t M = 4096;
    const size_t N = 4096;
    const size_t K = 4096;
    using T = half;

    std::unique_ptr<T[]> h_a = std::make_unique<T[]>(M * K);
    std::unique_ptr<T[]> h_b = std::make_unique<T[]>(N * K);
    T* h_c,*h_d ;
    T* A ,*B ,*C ,*D ;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f,1.0f);
    
    checkRuntime(cudaMalloc((void**)&A,M*K*sizeof(T)));
    checkRuntime(cudaMalloc((void**)&B,K*N*sizeof(T)));
    checkRuntime(cudaMalloc((void**)&C,M*N*sizeof(T)));
    checkRuntime(cudaMalloc((void**)&D,M*N*sizeof(T)));
    checkRuntime(cudaMallocHost((void**)&h_c,M*N*sizeof(T)));
    checkRuntime(cudaMallocHost((void**)&h_d,M*N*sizeof(T)));
    for(size_t i=0;i<M;++i){
        for(size_t j=0;j<K;++j){
            h_a[i*K+j]= __float2half(dis(gen));
            h_b[i*K+j]= __float2half(dis(gen));
            // h_a[i*K+j]= __float2half(1.0f);
            // h_b[i*K+j]= __float2half(1.0f);

        }
    }


    checkRuntime(cudaMemcpy(A,h_a.get(),M*K*sizeof(T),cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(B,h_b.get(),N*K*sizeof(T),cudaMemcpyHostToDevice));
    size_t smem_max_size = ( BM * BK + BN * BK ) * sizeof(half);
    printf("(M+BM -1 ) / BM = %d ,(N+BN -1) / BN = %d \n" ,(M+BM -1 ) / BM ,(N+BN -1) / BN);
    dim3 grid((M+BM -1 ) / BM ,(N+BN -1) / BN);
    dim3 block(128,1,1);
    // cudaFuncSetAttribute(mma_gemm_base, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max_size);
    dim3 block_size(32, 32, 1);
    for(int i=0;i<2;++i)
        cuda_hgemm<<<grid, block_size>>>(A, B, D, M, N, K);
    cudaDeviceSynchronize();
    mma_gemm_base<<< grid,block,smem_max_size>>> (A,B,C,M,N,K);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, C, M*N*sizeof(T), cudaMemcpyDeviceToHost);
    for(int i=0;i<16;++i){
        // if(i % 32 == 0)
        //     printf("\n");
       printf("%f ",__half2float(h_c[ i * N + i]));
    }
    printf("\n");

    
    // cuda_hgemm<<<grid, block_size>>>(A, B, D, M, N, K);
    // cudaDeviceSynchronize();

    // cudaMemcpy(h_d, D, M*N*sizeof(T), cudaMemcpyDeviceToHost);
    // for(int i=0;i<16;++i){
    //    printf("%f ",__half2float(h_d[ i * N + i]));
    //     // printBinary(*(uint16_t*)&h_c[i]);
    // }
    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    const __half alpha = __float2half(alpha_f);
    const __half beta = __float2half(beta_f);

    // 6. 调用 cublasHgemm (GEMM for __half)
    // 假设矩阵 A, B, C 均为列主序存储，且不进行转置 (CUBLAS_OP_N)
    // C = alpha * A * B + beta * C
    // 矩阵维度: C(M x N) = A(M x K) * B(K x N)
    
    const int ldA = M; // A 的 leading dimension: M
    const int ldB = K; // B 的 leading dimension: K
    const int ldC = M; // C 的 leading dimension: M

    printf("Executing cublasHgemm(C(%d x %d) = A(%d x %d) * B(%d x %d))\n", M, N, M, K, K, N);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasHgemm(
            handle,
            CUBLAS_OP_T, // op_A: 转置 A
            CUBLAS_OP_N, // op_B: 不转置 B
            M,           // m: 结果矩阵 C 的行数
            N,           // n: 结果矩阵 C 的列数
            K,           // k: 内积维度
            &alpha,      // alpha
            A,         // A 矩阵设备指针
            ldA,         // A 的 leading dimension
            B,         // B 矩阵设备指针
            ldB,         // B 的 leading dimension
            &beta,       // beta
            C,         // C 矩阵设备指针
            ldC          // C 的 leading dimension
        );

    cudaMemcpy(h_c, C, M*N*sizeof(T), cudaMemcpyDeviceToHost);
    for(int i=0;i<16;++i){
        printf("%f ",__half2float(h_c[i*N + i]));
        // printBinary(*(uint16_t*)&h_c[i]);
    }
        cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(kernelError));
        return -1;
    }
    printf("\nKernel launched successfully, waiting for completion...\n");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;

}