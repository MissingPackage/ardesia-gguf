#include "vendor/llama_cpp/common.cuh"
#include "vendor/llama_cpp/mma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace ggml_cuda_mma;

// C[16x8] = A[16x16] @ B[8x16]^T  (bf16 inputs, f32 accumulate), one m16n8k16.
__global__ void mma_test(const float* A, const float* B, float* C){
    __shared__ nv_bfloat162 As[16*8];   // 16 rows x 8 bf162 (=16 bf16 K)
    __shared__ nv_bfloat162 Bs[8*8];     // 8 rows(N) x 8 bf162 (=16 bf16 K)
    int t = threadIdx.x;
    __nv_bfloat16* Ab = (__nv_bfloat16*)As;
    __nv_bfloat16* Bb = (__nv_bfloat16*)Bs;
    for(int idx=t; idx<16*16; idx+=32){ Ab[idx]=__float2bfloat16(A[idx]); }
    for(int idx=t; idx<8*16;  idx+=32){ Bb[idx]=__float2bfloat16(B[idx]); }
    __syncwarp();
    tile<16,8,nv_bfloat162> tA;
    tile<8, 8,nv_bfloat162> tB;
    tile<16,8,float>        tD;   // zero-initialized
    #pragma unroll
    for(int l=0;l<tA.ne;l++) tA.x[l] = As[tA.get_i(l)*8 + tA.get_j(l)];
    #pragma unroll
    for(int l=0;l<tB.ne;l++) tB.x[l] = Bs[tB.get_i(l)*8 + tB.get_j(l)];
    mma(tD, tA, tB);
    #pragma unroll
    for(int l=0;l<tD.ne;l++) C[tD.get_i(l)*8 + tD.get_j(l)] = tD.x[l];
}
__global__ void ref(const float*A,const float*B,float*C){
    int i=blockIdx.x, j=threadIdx.x;   // 16 x 8
    float s=0; for(int k=0;k<16;k++) s+=__bfloat162float(__float2bfloat16(A[i*16+k]))*__bfloat162float(__float2bfloat16(B[j*16+k]));
    C[i*8+j]=s;
}
int main(){
    float *A,*B,*C,*Cr; cudaMallocManaged(&A,16*16*4); cudaMallocManaged(&B,8*16*4);
    cudaMallocManaged(&C,16*8*4); cudaMallocManaged(&Cr,16*8*4);
    srand(7); for(int i=0;i<256;i++)A[i]=(rand()/(float)RAND_MAX)*2-1; for(int i=0;i<128;i++)B[i]=(rand()/(float)RAND_MAX)*2-1;
    for(int i=0;i<128;i++)C[i]=-999;
    mma_test<<<1,32>>>(A,B,C); ref<<<16,8>>>(A,B,Cr);
    cudaError_t e=cudaDeviceSynchronize(); if(e){printf("ERR %s\n",cudaGetErrorString(e));return 1;}
    double mx=0; int bad=0; for(int i=0;i<128;i++){double d=fabs((double)C[i]-Cr[i]); if(d>mx)mx=d; if(d>1e-3){if(bad<4)printf("  [%d,%d] %f vs %f\n",i/8,i%8,C[i],Cr[i]);bad++;}}
    printf("max|abs|=%.3e bad=%d/128 -> %s\n",mx,bad, bad?"FAIL":"OK");
    return bad!=0;
}
