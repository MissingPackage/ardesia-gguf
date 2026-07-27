#include "ck/bf16_wmma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace torch_ggml_ops::ck;

// Fill fragments per the kernel contract, run the emulated WMMA, scatter acc.
__global__ void mma_kernel(const float* A, const float* B, float* C){
    int lane = threadIdx.x;            // one warp
    bf16_fragment fa, fb;
    __nv_bfloat16* a = fragment_data(fa);
    __nv_bfloat16* b = fragment_data(fb);
    int row = c_row(lane);             // lane & 15
    #pragma unroll
    for(int k=0;k<16;k++){
        a[k] = __float2bfloat16(A[row*16+k]);
        b[k] = __float2bfloat16(B[row*16+k]);
    }
    f32_accumulator acc;
    wmma_f32_16x16x16_bf16(acc, fa, fb);
    #pragma unroll
    for(int e=0;e<8;e++){
        int m = c_column(lane,e);      // 2e + lane>>4
        int n = c_row(lane);           // lane & 15
        C[m*16+n] = acc.values[e];
    }
}
// Reference: C[m][n] = sum_k bf16(A[m][k]) * bf16(B[n][k]), fp32 accumulate.
__global__ void ref_kernel(const float* A, const float* B, float* Cref){
    int m = blockIdx.x, n = threadIdx.x;   // 16x16
    float s=0.f;
    for(int k=0;k<16;k++){
        float av=__bfloat162float(__float2bfloat16(A[m*16+k]));
        float bv=__bfloat162float(__float2bfloat16(B[n*16+k]));
        s += av*bv;
    }
    Cref[m*16+n]=s;
}
int main(){
    const int N=256;
    float *A,*B,*C,*Cref;
    cudaMallocManaged(&A,N*sizeof(float)); cudaMallocManaged(&B,N*sizeof(float));
    cudaMallocManaged(&C,N*sizeof(float)); cudaMallocManaged(&Cref,N*sizeof(float));
    srand(1234);
    for(int i=0;i<N;i++){ A[i]=(rand()/(float)RAND_MAX)*2-1; B[i]=(rand()/(float)RAND_MAX)*2-1; C[i]=-999; }
    mma_kernel<<<1,32>>>(A,B,C);
    ref_kernel<<<16,16>>>(A,B,Cref);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("CUDA err: %s\n", cudaGetErrorString(e)); return 1; }
    double maxabs=0, maxrel=0; int bad=0;
    for(int i=0;i<N;i++){
        double d=fabs((double)C[i]-Cref[i]);
        double r=d/(fabs(Cref[i])+1e-6);
        if(d>maxabs)maxabs=d; if(r>maxrel)maxrel=r;
        if(d>1e-3 && r>1e-3){ if(bad<5) printf("  mismatch [%d,%d] C=%f ref=%f\n", i/16,i%16,C[i],Cref[i]); bad++; }
    }
    printf("max|abs|=%.3e  max rel=%.3e  bad=%d/%d\n", maxabs, maxrel, bad, N);
    printf("%s\n", bad==0 ? "WMMA EMULATION OK" : "WMMA EMULATION FAIL");
    return bad!=0;
}
