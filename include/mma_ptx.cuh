#pragma once

#include "config.cuh"


DEVICE static void ldmatrix_x1(uint32_t r0,uint32_t addr){
    asm volatile("ldmatrix.async.aligned.x1.m8n8.shared.b16 %0, [%1];\n"
                 : "=r"(r0)
                 : "r"(addr));
}

DEVICE static void ldmatrix_x2(uint32_t r0,uint32_t r1,uint32_t addr){
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1} [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

DEVICE static void ldmatrix_x4(uint32_t r0,uint32_t r1,uint32_t r2,uint32_t r3,uint32_t addr){
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3} [%4];\n"
                 : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3)
                 : "r"(addr)
    )
}

DEVICE static void hmma16816(uint32_t rd0, uint32_t rd1, uint32_t ra0, \
uint32_t ra1, uint32_t ra2, uint32_t ra3, uint32_t rb0, uint32_t rb1, uint32_t rc0, uint32_t rc1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
    : "=r"(rd0), "=r"(rd1)
    : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1), "r"(rc0), "r"(rc1));
}

DEVICE static void cp_async(void* dst, void* src, uint32_t size){
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1] , %2;\n"
    ::
    "r"(dst), "r"(src), "n"(size)
)
}

DEVICE static void cp_async_commit(){
    asm volatile("cp.async.commit_group;\n");
}

DEVICE static void cp_async_wait_group(uint32_t size){
    asm volatile("cp.async.wait_group %0;\n"
    :: "n"(size)
)
}

