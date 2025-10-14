#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void shared_address(){
    int tid = threadIdx.x;

    __shared__ float s_a[1024];

    if(tid == 0)
    printf("s_a[0] = %p, s_a[1] = %p\n",&s_a[0],&s_a[1]);
}

int main(){
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // 打印设备的各种共享内存信息
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Shared memory per SM: " << prop.sharedMemPerMultiprocessor << " bytes" << std::endl;
    std::cout << "Max shared memory per block (Hopper+): " << prop.sharedMemPerBlockOptin << " bytes" << std::endl;
    
    // 使用获取到的共享内存大小
    size_t shm_size = prop.sharedMemPerBlock;
    
    cudaFuncSetAttribute(shared_address,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, 
                        shm_size);
    shared_address<<<1,1024>>>();
    cudaDeviceSynchronize();
    return 0;
}