#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <config.cuh>
#include <cudaTypedefs.h>
#define HOST_DEVICE __host__ __device__ __forceinline__

#define bf16 __nv_bfloat16

#define GEMM_LIKELY(x) __builtin_expect(!!(x), 1)
#define GEMM_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define CHECK_CUDA(__call__)                                                                                \
    do{                                                                                                     \
        cudaError_t _err_ = _call_;                                                                     \
        if(GEMM_UNLIKELY(_err_ != cudaSuccess)) {                                                         \
            const char* _err_str_ = cudaGetErrorString(_err_);                                          \
            int _rt_version_ = 0;                                                                         \
            int _driver_version_ = 0;                                                                     \
            cudaRuntimeGetVersion(&_rt_version_);                                                         \
            cudaDriverGetVersion(&_driver_version_);                                                      \
            fprintf(stderr,"CUDA Runtime API error = %04d \"%s\", runtime version: %d, driver version: %d", \
                static_cast<int>(_err_), _err_str_, _rt_version_, _driver_version_);                        \
        }                                                                                                   \
    }while(0)                                                                                               \



template<int SShift,int BBits, int MBase>
struct Swizzle{
    static constexpr uint32_t num_s = SShift, num_b = BBits, num_m = MBase;
    static constexpr uint32_t SandM = num_s + num_m;
    static constexpr uint32_t SsubB = num_s - num_b;

    static constexpr uint32_t shift_msk = ((1U << num_s) - 1) << SandM;
    static constexpr uint32_t mid_msk = ((1U << SsubB) - 1) << SandM;
    static constexpr uint32_t base_msk = ((1U << num_b) - 1) << num_m;

    HOST_DEVICE static constexpr uint32_t  apply(uint32_t tid){
        uint32_t offset = (tid >> BBits << 7) + (((tid & ((1 << BBits) -1))) << MBase);
        uint32_t shift = (offset & shift_msk) >> SandM >> SsubB;
        uint32_t mid = (offset & mid_msk) >> SandM << (num_m + num_b);
        uint32_t base = (offset & base_msk) ^ (shift << num_m);
        return (((shift << SandM) + mid)) + base + (tid >> (SShift + BBits) << (SShift + BBits + MBase));
    }
    template <class Offset>
    HOST_DEVICE const auto operator()(Offset offset){
        return apply(static_cast<uint32_t>(offset));
    }

    HOST_DEVICE constexpr uint32_t operator()(uint32_t offset){
        return apply(offset);
    }
};



