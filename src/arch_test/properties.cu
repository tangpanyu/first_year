#include <cuda_runtime.h>
#include <iostream>

int main(){

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Device count: " << deviceCount << std::endl;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    std::cout << "Device name: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "multiProcessorCount(SM): "<< deviceProp.multiProcessorCount << std::endl;
    std::cout << "Memory clock rate: " << deviceProp.memoryClockRate << std::endl;
    std::cout << "Memory bus width: " << deviceProp.memoryBusWidth << std::endl;
    std::cout << "l2CacheSize: "<< (deviceProp.l2CacheSize >> 10) << "KB" << std::endl;
    std::cout << "persistingL2CacheMaxSize: " << (deviceProp.persistingL2CacheMaxSize >> 10) << "KB ,rate is :" << \
            float(deviceProp.persistingL2CacheMaxSize) /float(deviceProp.l2CacheSize) << std::endl;
    std::cout << "SharedMemPerMultiprocessor: " << (deviceProp.sharedMemPerMultiprocessor >> 10) << "KB" << std::endl;
    std::cout << "SharedMemPerBlockOptin: "<< (deviceProp.sharedMemPerBlockOptin >> 10) << "KB" << std::endl;
    std::cout << "sharedMemPerBlock: " << (deviceProp.sharedMemPerBlock >> 10) << "KB" <<std::endl;
    std::cout << "Total global memory: " << (deviceProp.totalGlobalMem >> 30) << "GB" << std::endl;
    std::cout << "maxBlocksPerMultiProcessor: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "maxGridSize: " << deviceProp.maxGridSize[0] << "x" << deviceProp.maxGridSize[1] << "x" << deviceProp.maxGridSize[2] << std::endl;
    double gpu_frequency = (double)deviceProp.clockRate * 1e3; // 
    std::cout << "GPU clock frequency: " << gpu_frequency << " Hz" << std::endl;
    

    return 0;
}