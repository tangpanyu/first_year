#include <cstdint>
#include <cstdio>
#include <iostream>
#include <cinttypes>
#include <sys/types.h>

#define HOST_DEVICE __host__ __device__ __forceinline__
template<int BBits, int MBase, int SShift>
struct Swizzle{
    static constexpr uint32_t num_s = BBits, num_b = MBase, num_m = SShift;
    static constexpr uint32_t SandM = num_s + num_m;
    static constexpr uint32_t SsubB = num_s - num_b;

    static constexpr uint32_t shift_msk = ((1U << num_s) - 1) << SandM;
    static constexpr uint32_t mid_msk = ((1U << SsubB) - 1) << SandM;
    static constexpr uint32_t base_msk = ((1U << num_b) - 1) << num_m;

    HOST_DEVICE static constexpr uint32_t  apply(uint32_t offset){
        uint32_t shift = (offset & shift_msk) >> SandM >> SsubB;
        uint32_t mid = (offset & mid_msk) >> SandM << (num_m + num_b);
        uint32_t base = (offset & base_msk) ^ (shift << num_m);
        return (((shift << SandM) + mid)) + base;
    }
    template <class Offset>
    HOST_DEVICE static const auto operator()(Offset offset){
        return apply(static_cast<uint32_t>(offset));
    }

    HOST_DEVICE static constexpr uint32_t operator()(uint32_t offset) const{
        return apply(offset);
    }
};

