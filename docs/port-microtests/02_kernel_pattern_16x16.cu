#include "vendor/llama_cpp/common.cuh"
#include "vendor/llama_cpp/mma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace ggml_cuda_mma;
// grad_input[16 x 16] = grad_output[16 x 16(K)] @ W[16(N) x 16(K)]^T
// A (grad_output) from GLOBAL via get_i/get_j; B (W) from SHARED via get_i/get_j; two N-halves.
__global__ void k(const float* GO, const float* W, float* GI){
    __shared__ __nv_bfloat16 Ws[16*16];         // N x K, bf16
    int t=threadIdx.x;
    for(int idx=t; idx<256; idx+=32) Ws[idx]=__float2bfloat16(W[idx]);
    __syncwarp();
    tile<16,8,nv_bfloat162> A;
    tile<8,8,nv_bfloat162>  B[2];
    tile<16,8,float>        D[2];
    // load A from global grad_output (row-major 16xK), K in bf16
    #pragma unroll
    for(int l=0;l<A.ne;l++){ int i=A.get_i(l), kc=A.get_j(l)*2;
        A.x[l]=__halves2bfloat162(__float2bfloat16(GO[i*16+kc]),__float2bfloat16(GO[i*16+kc+1])); }
    // load B halves from shared Ws (N x K)
    #pragma unroll
    for(int h=0;h<2;h++){
      #pragma unroll
      for(int l=0;l<B[h].ne;l++){ int n=h*8+B[h].get_i(l), kc=B[h].get_j(l)*2;
        B[h].x[l]=__halves2bfloat162(Ws[n*16+kc],Ws[n*16+kc+1]); }
    }
    mma(D[0],A,B[0]); mma(D[1],A,B[1]);
    // store natural layout: GI[m][n]
    #pragma unroll
    for(int h=0;h<2;h++){
      #pragma unroll
      for(int l=0;l<D[h].ne;l++){ int m=D[h].get_i(l), n=h*8+D[h].get_j(l); GI[m*16+n]=D[h].x[l]; }
    }
}
__global__ void ref(const float*GO,const float*W,float*GI){int m=blockIdx.x,n=threadIdx.x;float s=0;
    for(int kk=0;kk<16;kk++)s+=__bfloat162float(__float2bfloat16(GO[m*16+kk]))*__bfloat162float(__float2bfloat16(W[n*16+kk]));GI[m*16+n]=s;}
int main(){float*GO,*W,*GI,*GR;cudaMallocManaged(&GO,256*4);cudaMallocManaged(&W,256*4);cudaMallocManaged(&GI,256*4);cudaMallocManaged(&GR,256*4);
    srand(11);for(int i=0;i<256;i++){GO[i]=(rand()/(float)RAND_MAX)*2-1;W[i]=(rand()/(float)RAND_MAX)*2-1;GI[i]=-999;}
    k<<<1,32>>>(GO,W,GI);ref<<<16,16>>>(GO,W,GR);cudaError_t e=cudaDeviceSynchronize();if(e){printf("ERR %s\n",cudaGetErrorString(e));return 1;}
    double mx=0;int bad=0;for(int i=0;i<256;i++){double d=fabs((double)GI[i]-GR[i]);if(d>mx)mx=d;if(d>1e-3){if(bad<4)printf(" [%d,%d]%f vs %f\n",i/16,i%16,GI[i],GR[i]);bad++;}}
    printf("max|abs|=%.3e bad=%d/256 -> %s\n",mx,bad,bad?"FAIL":"OK");return bad!=0;}
