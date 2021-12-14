#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda.h"
#define t 0.01
#define l1 1
#define l2 1
#define BLOCK_SIZE 16
extern "C"
using namespace std;

struct matrix_node
{
        double x;
        double y;
};

__global__ void LU_Factorization(double * matrix_A ,double* matrix_U,double* matrix_L, int n);
void initialize_grid_points(matrix_node *grid, int m)
{
	for(int i=0; i<m; i++)
	{
		for(int j=0; j<m; j++)
		{
			grid[i*m+j].x = (i+1)*((double)1/(m+1));
			grid[i*m+j].y = (j+1)*((double)1/(m+1));
			//printf("%fl\n",grid[i*m+j].x);
                        
		}	
	}		
}

void construct_matrix_k( double *matrix_k, matrix_node* grid, int n)
{
	double x;
	double y;
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			x = (double)((grid[i].x - grid[j].x) * (grid[i].x - grid[j].x))/(2*(l1*l1));
			y = (double)((grid[i].y - grid[j].y) * (grid[i].y - grid[j].y))/(2*(l2*l2));
			matrix_k[i * n + j] = (1/sqrt((2*3.1415)))*exp(-(x+y));
		}
	}
	
}

void construct_matrix_k_pred(double * matrix_k_pred, matrix_node* grid, int n)
{
	double x;
	double y;
	for(int i=0; i<n; i++)
        {
        	x = (double)((0.2 - grid[i].x) * (0.2 - grid[i].x))/(2*(l1*l1));
                y = (double)((0.2 - grid[i].y) * (0.2 - grid[i].y))/(2*(l2*l2));
                matrix_k_pred[i] = (1/sqrt((2*3.1415)))*exp(-(x+y));
        }

}

void initialize_identity_matrix(double * identity_matrix, int n )
{
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			if(i==j)
			{	
				identity_matrix[i*n+j]=1*t;
			}
			else
			{	
			 	identity_matrix[i*n+j] = 0*t;
			}
		}
	}	
}

void compute_A( double* matrix_A, double* identity_matrix, double* matrix_k, int n )
{
	for( int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{	
			matrix_A[i*n+j] = identity_matrix[i*n+j]+matrix_k[i*n+j];
			//printf( " matrix_A value i*n+j is %d \n",matrix_A[i*n+j]);
		}
	}	
}


__global__ void LU_Factorization(double * matrix_A ,double* matrix_U,double* matrix_L, int n)
{
	//int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    	//int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    	//int tmp = 0;
    	//int idx;
        int index,index_3,index_2,index_4;
        int i,j,k;
        //matrix_U[gid][0] = 1;

        //printf("matrix A is %d \n", matrix_A[gid][0]);
    	for (int i = 0; i < n; i++)
    	{
        	// Upper Triangular
        	for (int k = i; k < n; k++)
        	{
            		// Summation of L(i, j) * U(j, k)
            		int sum = 0;
            		for (int j = 0; j < i; j++)
                	{
                		sum += (matrix_L[i*n+j] * matrix_U[j*n+k]);
                	}
			matrix_U[i*n+k] = matrix_A[i*n+k] - sum;
        	}	
 
        // Lower Triangular
        	for (int k = i; k < n; k++)
        	{
            		if (i == k)
                		matrix_L[i*n+i] = 1; // Diagonal as 1
            		else
            		{
                	// Summation of L(k, j) * U(j, i)
                		int sum = 0;
                		for (int j = 0; j < i; j++)
                    			sum += (matrix_L[k*n+j] * matrix_U[k*n+j]);
                // Evaluating L(k, i)
                		matrix_L[k*n+i] = (matrix_A[k*n+i] - sum) / matrix_U[i*n+i];
            		}
        	}
       
    }
  

}
void print_matrix (double* matrix, int size)
{
    printf("\n ---------------------  --------------------- \n");
    for(int i = 0; i < size; i++)
    {
        
        
            printf("%4.9lf ", matrix[i]);
        
        printf("\n");
    }
    printf("\n");
    printf("\n");
}

int main(int argc, char* argv[])
{
	int m;
	int n;
	m = atoi(argv[1]);
	n = m*m;
	matrix_node *grid;
	cudaMallocHost((void **) &grid, sizeof(matrix_node)*m*m);
	initialize_grid_points( grid,m); 
	
	// observed matrix n*1
	double *observed_points;
	cudaMallocHost((void **) &observed_points, sizeof(double)*n); 
	// k matrix_k n*n
	double *matrix_k;
	cudaMallocHost((void **) &matrix_k, sizeof(double)*(n*n));
	construct_matrix_k(matrix_k,grid,n);
	printf("matrix_k is \n");
	print_matrix(matrix_k,n*n);

	//k* n*1 matrix
	double *matrix_k_pred;
	cudaMallocHost((void **) &matrix_k_pred, sizeof(double)*n); 
	construct_matrix_k_pred(matrix_k_pred,grid,n);
	print_matrix(matrix_k_pred,n);

	// identity matrix 
	double * identity_matrix;
	cudaMallocHost((void **) &identity_matrix, sizeof(double)*(n*n));
	initialize_identity_matrix(identity_matrix,n);
	print_matrix(identity_matrix,n*n);
	
	// calculate matrix a which is n*n
	double * matrix_A;
        cudaMallocHost((void **) &matrix_A, sizeof(double)*(n*n));
        compute_A(matrix_A,identity_matrix, matrix_k,n);
	print_matrix(matrix_A,n*n);	

	// allocate matrix_U n*n for return from device to host
	double * matrix_U;
        cudaMallocHost((void **) &matrix_U, sizeof(double)*(n*n));
	//allocate matrix_L n*n for return from device to host
	double * matrix_L;
        cudaMallocHost((void **) &matrix_L, sizeof(double)*(n*n));	
	//LU_Factorization(matrix_A ,matrix_U,matrix_L, n);
	//print_matrix(matrix_U,n*n);
 	//print_matrix(matrix_L,n*n);


	// allocate device memory for matrix A n*n
	double * device_matrix_A;
	cudaMalloc((void **) &device_matrix_A, sizeof(double)*n*n);
	double * device_matrix_L;
        cudaMalloc((void **) &device_matrix_L, sizeof(double)*n*n);
	double * device_matrix_U;
        cudaMalloc((void **) &device_matrix_U, sizeof(double)*n*n);
	cudaMemcpy(device_matrix_A, matrix_A, sizeof(double)*n*n, cudaMemcpyHostToDevice);
	
	unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    	unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    	dim3 dimGrid(grid_cols, grid_rows);
    	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	cudaMemcpy(matrix_U, device_matrix_A, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
	//LU_Factorization(devic_matrix_A ,device_matrix_U, device_matrix_L,n)
	//cudaThreadSynchronize();
	//print_matrix(matrix_A,n*n);
	printf("copy device matrix a back to host \n"); 
	print_matrix(matrix_U,n*n);
}

