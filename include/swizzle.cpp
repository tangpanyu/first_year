#include <cstdint>
#include <cstdio>
#include <iostream>
#include <cinttypes>

#ifdef debug
    #define debug(marco) marco;
#endif

uint32_t num_s =3, num_b = 0, num_m = 4;
uint32_t SandM = num_s + num_m;
uint32_t SsubB = num_s - num_b;
// 打印uint32_t数值的低8位二进制表示
void print_binary_low8(uint32_t value) {
    printf("0x");
    for (int i = 9; i >=0; i--) {
        if (i % 3 == 2)
            printf(",");
        printf("%" PRIu64, (value >> i) & 1);
    }
    printf("\n");
}
    // 先算出在需要swizzle的原矩阵中的偏移映射到smem中，也就是说<3,2，4>每个矩阵都只映射到共享内存的前4列，后面的4列不用管，因为是填充的。
    // uint32_t base = (tid >> BBits << 7) + (((tid & ((1 << BBits) -1))) << MBase);  //先算出原行，再算出列
    // 加的偏移和shift和bit的表示范围有关，和tid无关。
// 元素的位置不影响，列的高位由行的低位决定，因为要填补一行，行的位置由行决定，一行填补完了在继续下一行做swizzle，填补的行不需要改变当前行的swizzle,所以
// 先决定出行号，再决定出列号，再做swizzle.
uint32_t apply(uint32_t offset) {

    uint32_t shift_msk = ((1ULL << num_s) - 1) << SandM; // 0x1,111,000 num_s其实是总列数，行数不会大于总列数，所以问题不大,
    
    uint32_t mid_msk = ((1ULL << SsubB) - 1) << SandM;

    uint32_t base_msk = ((1ULL << num_b) - 1) << num_m; // 0x0,00,111
    
    uint32_t shift = (offset & shift_msk) >> SandM >> SsubB;  // 得到s分块,只包括行低位和列高位。
    // 得到行和列的中间值  计算列的高位，列的高位是由行的低位决定的，和B的数据没关系。
    uint32_t mid = (offset & mid_msk) >> SandM << (num_m + num_b);
    // 得到b分块 ： 只包括列号的低位,但是他是不变的，变得只有列高位和行所有位置。
    uint32_t base = (offset & base_msk) ^ (shift << num_m);
    uint32_t ret = (((shift << SandM) + mid))  + base;
    // printf("shift_msk = ");
    // print_binary_low8(shift_msk);
    // printf("mid_msk = ");
    // print_binary_low8(mid_msk);
    // printf("base_msk = ");
    // print_binary_low8(base_msk);
    // printf("shift = ");
    // print_binary_low8(shift);
    // printf("mid = ");
    // print_binary_low8(mid);
    // printf("base = ");
    // print_binary_low8(base);
    // printf("ret = ");
    // print_binary_low8(ret);
    return ret; // 避免编译警告
}

int main(){
    const int M=16;
    const int K=128;
    const int BM=64;
    const int BK=64;
    const int Swizzle=2;
    uint32_t * h_a = new uint32_t[M*K];
    uint32_t* d_a, *d_c;

    for(uint32_t i=0;i<M*K;i++){
        h_a[i] = apply(i);
        // printf("%3" PRIu64" ", h_a[i]); 
    }
    for(int i=0;i<M;i++){
        if(i!=0)
            printf("\n");
        for(int j=0;j<K;j+=16){
            uint32_t offset = h_a[i*K+j];
            printf("%3" PRIu64" ", offset/16); 
        } 
    }
    printf("\n");
    // uint32_t offset = 130;
    // printf("offset: ");
    // print_binary_low8(offset);
    // offset = apply(offset);
    // printf("resutl  = %3" PRIu64" \n", offset/16); 

    return 0;
}