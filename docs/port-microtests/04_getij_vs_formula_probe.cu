#include "vendor/llama_cpp/common.cuh"
#include "vendor/llama_cpp/mma.cuh"
#include <cstdio>
using namespace ggml_cuda_mma;
__global__ void k(){
    int t=threadIdx.x, lane=t&31;
    tile<16,8,nv_bfloat162> A; tile<8,8,nv_bfloat162> B; tile<16,8,float> D;
    if(t==5){
      for(int l=0;l<A.ne;l++) printf("A l=%d  get_i=%d formI=%d | get_j=%d formJ=%d\n",l,A.get_i(l),(l/2)*8+(lane>>2),A.get_j(l),(lane&3)*2+(l&1));
      for(int l=0;l<B.ne;l++) printf("B l=%d  get_i=%d formI=%d | get_j=%d formJ=%d\n",l,B.get_i(l),(lane>>2),B.get_j(l),l*4+(lane&3));
      for(int l=0;l<D.ne;l++) printf("D l=%d  get_i=%d formI=%d | get_j=%d formJ=%d\n",l,D.get_i(l),(l/2)*8+(lane>>2),D.get_j(l),(lane&3)*2+(l&1));
    }
}
int main(){k<<<1,32>>>();cudaDeviceSynchronize();return 0;}
