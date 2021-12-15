#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda.h"
#define t 0.01
#define l1 1
#define l2 1
#define BLOCK_SIZE 2
extern "C"
using namespace std;

struct matrix_node
{
        double x;
        double y;
};

__global__ void LU_Factorization(double * matrix_A ,double* matrix_U,double* matrix_L, int n);
__global__ void calculate_y(double * matrix_f, double * matrix_y, double * matrix_L,int n);
//__global__ void calculate_x(double * matrix_y, double * matrix_x,double * matrix_U,int n);
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

void assign_observed_value( double * matrix, matrix_node* grid, int m)
{
	int index = 0; 
	for( int i=0; i<m; i++)
	{
		for(int j=0; j<m; j++)			
		{
			//printf("%lf\n",((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05);
		    matrix[index] = 1 - (((grid[i*m+j].x - 0.5)*(grid[i*m+j].x - 0.5)) + ((grid[i*m+j].y - 0.5)*(grid[i*m+j].y-0.5))) + (((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05);
			index++;	
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
	int row = blockIdx.y *   BLOCK_SIZE + threadIdx.y;
    	int col = blockIdx.x *   BLOCK_SIZE + threadIdx.x;
    	//printf(" row is %d\n " , row);
	//printf(" col  is %d \n" , col);
	int tmp = 0;
    	int idx;
        //int index,index_3,index_2,index_4;
        int i,j,k;
        //matrix_U[gid][0] = 1;


	
        //printf("matrix A is %d \n", matrix_A[gid][0]);
    	for (int i = 0; i < n; i++)
    	{
        	// Upper Triangular
		if ( row >=i && row <n)
		{	
			printf(" row is %d\n " , row);
        		printf(" col  is %d \n" , col);
			int sum=0;
			if (col < i)
			{	
				sum += ( matrix_L[i*n+col] * matrix_U[col*n+row]);
			}
			matrix_U[i*n+row] = matrix_A[i*n+col]-sum;
			__syncthreads(); 
       
            		if (i == row)
			{
                		matrix_L[i*n+i] = 1; // Diagonal as 1
            		}
			else
            		{
                	// Summation of L(k, j) * U(j, i)
                		int sum = 0;
                		if( col <i)
				{

                    			sum += (matrix_L[row*n+col] * matrix_U[row*n+col]);
                		}
                		matrix_L[row*n+i] = (matrix_A[row*n+i] - sum) / matrix_U[i*n+i];
            		}
		
        	}

       
    	}
  
	
}

__global__ void calculate_y(double * matrix_f, double * matrix_y, double * matrix_L,int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
	int i,j,index;
	printf(" y row is %d %d \n " , row, threadIdx.y);
        printf(" y col  is %d \n" , col);
	if ( row < n)
	{	
		matrix_y[row]=matrix_f[row];
		if(col < row )
		{
        		matrix_y[row]-=matrix_L[row*n+col]*matrix_y[col];
       	 	}
		__syncthreads();	
	}
    	
}

void calculate_x(double * matrix_y, double * matrix_x,double * matrix_U,int n)
{
        int i,j;
    for(i=n-1; i>=0; i--)
    {
        matrix_x[i]=matrix_y[i];
        for(j=i+1; j<n; j++)
        {
            matrix_x[i]-=matrix_U[i*n+j]*matrix_x[j];
        }
	matrix_x[i]/=matrix_U[i];
    }
}
/*
__global__ void calculate_x(double * matrix_y, double * matrix_x,double * matrix_U,int n)
{
        int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
        int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
        matrix_x[row]=matrix_y[row];
        if(col>row)
        {	
			
            matrix_x[row]-=matrix_U[row*n+col]*matrix_x[col];
       	__syncthreads();
	} 
	
	matrix_x[row]/=matrix_U[row*n+row];
}
*/

void transpose(double * matrix_k_pred, double * matrix_k_pred_transpose, int r,int c )
{
 for (int i = 0; i < r; ++i)
   for (int j = 0; j < c; ++j) 
 {
     matrix_k_pred_transpose[j*r+i] = matrix_k_pred[i*c+j];
   }
}


void multiplyMatrices( double * matrix1,
                      double * matrix2,
                      double * result,
                      int r1, int c1, int r2, int c2) {

   // Initializing elements of matrix mult to 0.
   for (int i = 0; i < r1; ++i) {
      for (int j = 0; j < c2; ++j) {
         result[i*c2+j] = 0;
      }
   }

   // Multiplying first and second matrices and storing it in result
   for (int i = 0; i < r1; ++i) {
      for (int j = 0; j < c2; ++j) {
         for (int k = 0; k < c1; ++k) {
            result[i*c2+j] += matrix1[i*c1+k] * matrix2[k*c2+j];
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
	assign_observed_value( observed_points, grid,  m);
	
	double *device_observed_points;
        cudaMalloc((void **) &device_observed_points, sizeof(double)*n);
	cudaMemcpy(device_observed_points, observed_points, sizeof(double)*n, cudaMemcpyHostToDevice);
	printf(" print observed points \n");	
	print_matrix(observed_points,n);

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


	// allocate device memory for matrix A n*n
	double * device_matrix_A;
	cudaMalloc((void **) &device_matrix_A, sizeof(double)*n*n);
	double * device_matrix_L;
        cudaMalloc((void **) &device_matrix_L, sizeof(double)*n*n);
	double * device_matrix_U;
        cudaMalloc((void **) &device_matrix_U, sizeof(double)*n*n);
	cudaMemcpy(device_matrix_A, matrix_A, sizeof(double)*n*n, cudaMemcpyHostToDevice);
	
	// we use one block ezch block is n*n 
	
//	unsigned int grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  //  	unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    //	dim3 dimGrid(grid_cols, grid_rows);
    	dim3 dimBlock(n, n);
	
	LU_Factorization<<<m, dimBlock>>>(device_matrix_A ,device_matrix_U, device_matrix_L,n);
	cudaMemcpy(matrix_L, device_matrix_L, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(matrix_U, device_matrix_U, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
	//cudaThreadSynchronize();
	
	printf("copy device matrix L a back to host \n"); 
	print_matrix(matrix_L,n*n);
	printf("copy device matrix U a back to host \n");
        print_matrix(matrix_U,n*n);


	// allocate matrix_y for Ly=b y = n*1
	double *matrix_y;
        cudaMallocHost((void **) &matrix_y, sizeof(double)*n);
	double *device_matrix_y;
        cudaMalloc((void **) &device_matrix_y, sizeof(double)*n);

	//calculate matrix y 
	calculate_y<<<m, dimBlock>>>(device_observed_points, device_matrix_y,device_matrix_L, n);
	cudaMemcpy(matrix_y, device_matrix_y, sizeof(double)*n, cudaMemcpyDeviceToHost);
	print_matrix(matrix_y,n);


	//allocate matrix x Ux=y x=n*1
	double *matrix_x;
        cudaMallocHost((void **) &matrix_x, sizeof(double)*n);
	//double *device_matrix_x;
        //cudaMalloc((void **) &device_matrix_x, sizeof(double)*n);
	//calculate matrix x 
	//calculate_x<<<1,dimBlock>>>(device_matrix_y,device_matrix_x, device_matrix_U, n);
	//cudaMemcpy(matrix_x, device_matrix_x, sizeof(double)*n, cudaMemcpyDeviceToHost);
	
	calculate_x( matrix_y,  matrix_x, matrix_U,n);
	print_matrix(matrix_x,n);
	// allocate for matrix k predict transpose 
	double *matrix_k_pred_transpose;
        cudaMallocHost((void **) &matrix_k_pred_transpose, sizeof(double)*n);
	transpose(matrix_k_pred, matrix_k_pred_transpose,n,1);
	print_matrix(matrix_k_pred_transpose,n);	
	double *result;
        cudaMallocHost((void **) &result, sizeof(double)*1);

	multiplyMatrices(matrix_k_pred_transpose,matrix_x,result,1,n,n,1);
	print_matrix(result,1);
}

