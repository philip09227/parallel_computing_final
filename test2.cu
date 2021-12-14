#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda.h"
#define N 16

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
if (i < N && j < N)
C[i][j] = A[i][j] + B[i][j];
}

int main()
{
float A[N][N];
float B[N][N];
float C[N][N];

float (*d_A)[N]; //pointers to arrays of dimension N
float (*d_B)[N];
float (*d_C)[N];

for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
        A[i][j] = i;
        B[i][j] = j;
    }
}       

//allocation
cudaMalloc((void**)&d_A, (N*N)*sizeof(float));
cudaMalloc((void**)&d_B, (N*N)*sizeof(float));
cudaMalloc((void**)&d_C, (N*N)*sizeof(float));

//copying from host to device
cudaMemcpy(d_A, A, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_C, C, (N*N)*sizeof(float), cudaMemcpyHostToDevice);

// Kernel invocation
dim3 threadsPerBlock(16, 16);
dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

//copying from device to host
cudaMemcpy(A, (d_A), (N*N)*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(B, (d_B), (N*N)*sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(C, (d_C), (N*N)*sizeof(float), cudaMemcpyDeviceToHost);


for(int i=0;i<N;i++)
      for(int j=0;j<N;j++)
      {
        printf("%d\n",C[i][j]);

      }

}
