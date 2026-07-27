#include "vendor/llama_cpp/common.cuh"
#include "vendor/llama_cpp/mma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace ggml_cuda_mma;
// 128-thread block = 4 warps, each computes GI[warp] = GO @ W^T (16x16) using lane=tid%32.
__global__ void k(const float* GO, const float* W, float* GI){
    __shared__ __nv_bfloat16 Ws[16*16];
    int t=threadIdx.x, lane=t&31, warp=t>>5;
    for(int idx=t; idx<256; idx+=128) Ws[idx]=__float2bfloat16(W[idx]);
    __syncthreads();
    tile<16,8,nv_bfloat162> A; tile<8,8,nv_bfloat162> B[2]; tile<16,8,float> D[2];
    #pragma unroll
    for(int l=0;l<A.ne;l++){ int i=(l&1)*8+(lane>>2), kc=2*((l/2)*4+(lane&3));
        A.x[l]=__halves2bfloat162(__float2bfloat16(GO[i*16+kc]),__float2bfloat16(GO[i*16+kc+1])); }
    #pragma unroll
    for(int h=0;h<2;h++)
      #pragma unroll
      for(int l=0;l<B[h].ne;l++){ int n=h*8+(lane>>2), kc=2*(l*4+(lane&3));
        B[h].x[l]=__halves2bfloat162(Ws[n*16+kc],Ws[n*16+kc+1]); }
    mma(D[0],A,B[0]); mma(D[1],A,B[1]);
    #pragma unroll
    for(int h=0;h<2;h++)
      #pragma unroll
      for(int l=0;l<D[h].ne;l++){ int m=(l/2)*8+(lane>>2), n=h*8+((lane&3)*2+(l&1));
        GI[warp*256 + m*16+n]=D[h].x[l]; }
}
__global__ void ref(const float*GO,const float*W,float*GR){int m=blockIdx.x,n=threadIdx.x;float s=0;
    for(int kk=0;kk<16;kk++)s+=__bfloat162float(__float2bfloat16(GO[m*16+kk]))*__bfloat162float(__float2bfloat16(W[n*16+kk]));GR[m*16+n]=s;}
int main(){float*GO,*W,*GI,*GR;cudaMallocManaged(&GO,256*4);cudaMallocManaged(&W,256*4);cudaMallocManaged(&GI,4*256*4);cudaMallocManaged(&GR,256*4);
    srand(3);for(int i=0;i<256;i++){GO[i]=(rand()/(float)RAND_MAX)*2-1;W[i]=(rand()/(float)RAND_MAX)*2-1;}for(int i=0;i<4*256;i++)GI[i]=-999;
    k<<<1,128>>>(GO,W,GI);ref<<<16,16>>>(GO,W,GR);cudaError_t e=cudaDeviceSynchronize();if(e){printf("ERR %s\n",cudaGetErrorString(e));return 1;}
    int bad=0;double mx=0;for(int w=0;w<4;w++)for(int i=0;i<256;i++){double d=fabs((double)GI[w*256+i]-GR[i]);if(d>mx)mx=d;if(d>1e-3){if(bad<4)printf(" w%d[%d,%d]%f vs %f\n",w,i/16,i%16,GI[w*256+i],GR[i]);bad++;}}
    printf("4 warps: max|abs|=%.3e bad=%d/1024 -> %s\n",mx,bad,bad?"FAIL":"OK");return bad!=0;}
