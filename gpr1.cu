#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cuda.h"

extern "C"
using namespace std;


#define t 0.01
#define l1 1
#define l2 1
#define n 4
__global__ void LU_Factorization(double matrix_U[16][0] ,int n); //double ** matrix_A ,double** matrix_U,double** matrix_L, int n);

struct matrix_node
{
	double x;
	double y;
};


// allocate matrix in host memory 
double** allocate_matrix( int row, int col)
{	
	double **matrix = NULL;
    	matrix = (double**) calloc (row, sizeof(double*));
    	for(int i = 0; i < row; i++)
   	 {	
        	matrix[i] = (double*) calloc (col, sizeof(double));
    	}
    	return matrix;
}

void assign_observed_value( double ** matrix, matrix_node* grid, int m)
{
	int index = 0; 
	for( int i=0; i<m; i++)
	{
		for(int j=0; j<m; j++)			
		{
			//printf("%lf\n",((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05);
		    matrix[index][0] = 1 - (((grid[i*m+j].x - 0.5)*(grid[i*m+j].x - 0.5)) + ((grid[i*m+j].y - 0.5)*(grid[i*m+j].y-0.5))) + (((double)rand()/(10.0 * (double)RAND_MAX)) - 0.05);
			index++;	
		}
	} 
}

// initialize grid with value from 1/m to  (m-1)/m
void initialize_grid_points(matrix_node *grid, int m)
{
	int index=0;
	for(int i=1; i<=m; i++)
	{
		for(int j=1; j<=m; j++)
		{
			grid[index].x = i*((float)1/(m+1));
			grid[index].y = j*((float)1/(m+1));
			//printf("%lf\n",i*((float)1/(m+1)));
			index++;
                        printf("%u\n",index);
		}	
	}
	printf("%f\n",index);		
}
// k is a n*n matrix K(a,b) a =(x1,y1) b= (x2,y2) .... representivly 
void construct_matrix_k( double **matrix_k, matrix_node* grid, int n)
{
	double x;
	double y;
	int index;
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			
			x = (double)((grid[i].x - grid[j].x) * (grid[i].x - grid[j].x))/(2*(l1*l1));
			y = (double)((grid[i].y - grid[j].y) * (grid[i].y - grid[j].y))/(2*(l2*l2));
			index = i*n+j;	
			matrix_k[index][0] = (1/sqrt((2*3.1415)))*exp(-(x+y));
		}
	}
	
}

void construct_matrix_k_pred(double ** matrix_k_pred, matrix_node* grid, int n)
{
	double x;
	double y;
	for(int i=0; i<n; i++)
        {
        	x = (double)((0.2 - grid[i].x) * (0.2 - grid[i].x))/(2*(l1*l1));
                y = (double)((0.2 - grid[i].y) * (0.2 - grid[i].y))/(2*(l2*l2));
                matrix_k_pred[i][0] = (1/sqrt((2*3.1415)))*exp(-(x+y));
        }

}

void initialize_identity_matrix(double ** identity_matrix, int n)
{
	int index;
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{	
			index = i*n+j;
			if(i==j)
			{	
				identity_matrix[index][0]=1*t;
			}
			else
			{	
			 	identity_matrix[index][0] = 0*t;
			}
		}
	}	
}


void compute_A( double** matrix_A, double** identity_matrix, double** matrix_k, int n)
{
	int index; 
	for( int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{	
			index = i*n+j;
			matrix_A[index][0] = identity_matrix[index][0]+matrix_k[index][0];
		}
	}	
}

// inverse matrix A-1 = adjA/|A|
//adjA = 
//void inverse_matrix()

/*
void cholesky_compute(int n, double ** matrix, VECTOR p)
{
    int i,j,k;
    double sum;

//#pragma omp for
    for (i = 0; i < n; i++)
    {
        for (j = i; j < n; j++)
        {
            sum = matrix[i][j];
            for (k = i - 1; k >= 0; k--)
            {
                sum -= matrix[i][k] * matrix[j][k];
            }
            if (i == j)
            {
                if (sum <= 0)
                {
                    printf(" a is not positive definite!\n");
                }
                p[i] = sqrt(sum);
            }
            else
            {
                matrix[j][i] = sum / p[i];
            }
        }
    }
}


void decompose_and_get_inverse (int n, double ** input_matrix, double ** output_matrix)
{
    int i,j,k;
    double sum;
    VECTOR p;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            output_matrix[i][j] = input_matrix[i][j];

    cholesky_compute (n, output_matrix, p);

//#pragma omp for
    for (i = 0; i < n; i++)
    {
        output_matrix[i][i] = 1 / p[i];
        for (j = i + 1; j < n; j++)
        {
            sum = 0;
            for (k = i; k < j; k++)
            {
                sum -= output_matrix[j][k] * output_matrix[k][i];
            }
            output_matrix[j][i] = sum / p[j];
        }
    }
}

void get_inverse_by_cholesky (int n, double ** input_matrix, double ** output_matrix)
{
    int i,j,k;
    decompose_and_get_inverse (n, input_matrix, output_matrix);
//#pragma omp for
    for (i = 0; i < n; i++)
    {
        for (j = i + 1; j < n; j++)
        {
            output_matrix[i][j] = 0.0;
        }
    }
    for (i = 0; i < n; i++)
    {
        output_matrix[i][i] *= output_matrix[i][i];
        for (k = i + 1; k < n; k++)
        {
            output_matrix[i][i] += output_matrix[k][i] * output_matrix[k][i];
        }
        for (j = i + 1; j < n; j++)
        {
            for (k = j; k < n; k++)
            {
                output_matrix[i][j] += output_matrix[k][i] * output_matrix[k][j];
            }
        }
    }
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < i; j++)
        {
            output_matrix[i][j] = output_matrix[j][i];
        }
    }
}

void get_transpose (double ** output, double **input, int n)
{

    for(int i = 0; i < n; i++)
    {
        output[i][0] = input[0][i];
    }
}

double** run_solver (double **K_inverse, double **k, double **f_train, int train, int test)
{
{
    double **product = alloc_multidim_matrix (train , test);
//#pragma omp task default (none) shared(product, K_inverse, k, train, test)
    multiply_matrix (product, K_inverse, k, train , train, train, test);

    double **output = alloc_multidim_matrix (1, test);
//#pragma omp task default (none) shared(output, f_train, product, train, test)
    multiply_matrix (output, f_train, product, 1, train, train, test);

    return output;

}

*/


__global__  void LU_Factorization( double matrix_U[16][0],int n )//double ** matrix_A ,double** matrix_U,double** matrix_L, int n)

{
	//int row = blockIdx.y * blockDim.y + threadIdx.y; 
	//int col = blockIdx.x * blockDim.x + threadIdx.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int gid = n * y + x;
	printf( " gid is %d \n",gid); 
	//int gid = blockDim.x * blockIdx.x + threadIdx.x;
	//double * h_data = (double *)malloc(sizeof(double));
	//cudaMemcpy(h_data, matrix_U[gid],sizeof(double) , cudaMemcpyDeviceToHost);
//	printf(" %d ", *h_data);

	int index,index_3,index_2,index_4;
	int i,j,k;
	matrix_U[gid][0] = 1;
/*
	//printf("matrix A is %d \n", matrix_A[gid][0]);
    for (int i = 0; i < n; i++)
    {
        // Upper Triangular
        for (int k = i; k < n; k++)
        {
            // Summation of L(i, j) * U(j, k)
            int sum = 0;
            for (int j = 0; j < i; j++)
                
		//sum += (matrix_L[i][j] * matrix_U[j][k]);
 		sum += (matrix_L[i*n+j][0] * matrix_U[j*n+k][0]);
            // Evaluating U(i, k)
            //matrix_U[i][k] = matrix_A[i][k] - sum;
		matrix_U[i*n+k][0] = matrix_A[i*n+k][0] - sum;
        }
 
        // Lower Triangular
        for (int k = i; k < n; k++)
        {
            if (i == k)
                matrix_L[i*n+i][0] = 1; // Diagonal as 1
            else
            {
                // Summation of L(k, j) * U(j, i)
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (matrix_L[k*n+j][0] * matrix_U[k*n+j][0]);
 
                // Evaluating L(k, i)
                matrix_L[k*n+i][0] = (matrix_A[k*n+i][0] - sum) / matrix_U[i*n+i][0];
            }
        }
	__syncthreads();
    }
//	print_matrix(matrix_U,n*n,1);	
*/
}


void calculate_y(double ** matrix_f, double ** matrix_y,double ** matrix_L,int n)
{
	int i,j,index;
    for(i=0; i<n; i++)
    {
	
        matrix_y[i][0]=matrix_f[i][0];
        for(j=0; j<i; j++)
        {
	    index = i*n+j;
            matrix_y[i][0]-=matrix_L[index][0]*matrix_y[j][0];
        }
    }
}

void calculate_x(double ** matrix_y, double ** matrix_x,double ** matrix_U,int n)
{
        int i,j,index,index_2;
    for(i=n-1; i>=0; i--)
    {
        matrix_x[i][0]=matrix_y[i][0];
        for(j=i+1; j<n; j++)
        {
		index = i*n+j;
            matrix_x[i][0]-=matrix_U[index][0]*matrix_x[j][0];
        }
	index_2 = i*n+i;
	matrix_x[i][0]/=matrix_U[index_2][0];
    }
}


void transpose(double ** matrix_k_pred, double ** matrix_k_pred_transpose, int r,int c )
{
	for (int i = 0; i < r; ++i)
  	for (int j = 0; j < c; ++j) 
	{
    	matrix_k_pred_transpose[j][i] = matrix_k_pred[i][j];
  	}
}


void multiplyMatrices( double ** matrix1,
                      double ** matrix2,
                      double ** result,
                      int r1, int c1, int r2, int c2) {

   // Initializing elements of matrix mult to 0.
   for (int i = 0; i < r1; ++i) {
      for (int j = 0; j < c2; ++j) {
         result[i][j] = 0;
      }
   }

   // Multiplying first and second matrices and storing it in result
   for (int i = 0; i < r1; ++i) {
      for (int j = 0; j < c2; ++j) {
         for (int k = 0; k < c1; ++k) {
            result[i][j] += matrix1[i][k] * matrix2[k][j];
         }
      }
   }
}



void print_matrix (double** matrix, int row, int col)
{
    printf("\n ---------------------  --------------------- \n");
    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            printf("%4.9lf ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
}

void print_points (matrix_node *grid_points, int m)
{
    int id = 0;
    printf("\n --------------------- Printing grid coordinates --------------------- \n");
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < m; j++)
        {
            printf("(%lf %lf) ", grid_points[id].x, grid_points[id].y);
            id++;
        }
        printf("\n");
    }

}




int main(int argc, char* argv[])
{
	int m;
	int n;
	m = atoi(argv[1]);
	//n=m*m;
	// grid 1/m+1
	matrix_node * grid = (matrix_node*)malloc(n*sizeof(matrix_node)); 
	// allocate grid on device 
	matrix_node *device_grid;
	cudaMalloc(&device_grid, n*sizeof(matrix_node));      
	// initialize value of grid on host
	initialize_grid_points(grid,m);
	// copy into device 
	cudaMemcpy(device_grid, grid, n*sizeof(matrix_node),cudaMemcpyHostToDevice);
	print_points(grid,m);

	

	// observed matrix on host n*1
	double** matrix = allocate_matrix(n,1);
	assign_observed_value( matrix, grid,m);
	print_matrix(matrix,n,1);
	
	
	// k n*n matrix
	double** matrix_k = allocate_matrix(n*n,1);
	construct_matrix_k(matrix_k,grid,n);
	print_matrix(matrix_k,n*n,1);
	
	// k* pred n*1 matrix
	double** matrix_k_pred = allocate_matrix(n,1);
	construct_matrix_k_pred(matrix_k_pred,grid,n);
	print_matrix(matrix_k_pred,n,1);
	
	// allocate matrix in device memory 
	// allocate device memovry for observed points value which is n*1
	double ** device_observed_points;
	cudaMalloc(&device_observed_points, n*sizeof(double));
	cudaMemcpy(device_observed_points,matrix, n*sizeof(double),cudaMemcpyHostToDevice);
	
	//initialise n*n identity matrix
	double ** identity_matrix = allocate_matrix(n*n,1);
	initialize_identity_matrix(identity_matrix,n);
	print_matrix(identity_matrix,n*n,1);	
	
	//cudaMemcpy(device_grid, grid, (n+1)*sizeof(matrix_node),cudaMemcpyHostToDevice);
	// computer A	
	double ** matrix_A = allocate_matrix(n*n,1);	
	compute_A( matrix_A, identity_matrix, matrix_k,n);
	printf("host matrix A  is : \n");
	print_matrix(matrix_A,n*n,1);

	double ** matrix_U = allocate_matrix(n*n,1);;
	double ** matrix_L = allocate_matrix(n*n,1);
	double matrix_temp[16][0];
	
	double ** device_matrix_A;
        cudaMalloc( &device_matrix_A, (n*n)*sizeof(double));
        cudaMemcpy(device_matrix_A,matrix_A, (n*n)*sizeof(double),cudaMemcpyHostToDevice);
	
        //double *matrix_temp  = (double *)malloc(1*sizeof(double));
//        cudaMemcpy(matrix_temp, device_matrix_A ,(n*n)*sizeof(double) , cudaMemcpyDeviceToHost);
//	printf( "first item in matrixa is %d \n ", matrix_temp);	
	
	double  device_matrix_U[16][0];
        //cudaMalloc( &device_matrix_U, n*n*sizeof(double));
        //cudaMemcpy(device_matrix_U,matrix_U, n*n*sizeof(double),cudaMemcpyHostToDevice);
	double ** device_matrix_L;
        cudaMalloc( &device_matrix_L, n*n*sizeof(double));
        cudaMemcpy(device_matrix_L,matrix_L, n*n*sizeof(double),cudaMemcpyHostToDevice);
	//double  device_matrix_temp[16][0];
	//cudaMalloc( &device_matrix_temp, n*n*sizeof(double));
	
	//LU_Factorization(matrix_A , matrix_U,matrix_L,n);
	//dim3 blocks1D(1);	
	//int threadsPerBlock = 1;
  	//int blocksPerGrid =1;
	int threadsPerBlock = n;//256;
   	int blocksPerGrid =n;//(n + threadsPerBlock - 1) / threadsPerBlock;
	LU_Factorization<<<blocksPerGrid, threadsPerBlock>>>( device_matrix_U,n)//device_matrix_A, device_matrix_U,device_matrix_L, n);
	cudaMemcpy(matrix_temp, device_matrix_U, 16*sizeof(double), cudaMemcpyDeviceToHost);
	printf("AFTER LU is : \n");
	print_matrix(matrix_temp,n*n,1);
/*	
	print_matrix(matrix_L,n*n,1);
	print_matrix(matrix_U,n*n,1);

	double ** matrix_x = allocate_matrix(n,1);
        double ** matrix_y = allocate_matrix(n,1);
	calculate_y(matrix, matrix_y,matrix_L, n);
	calculate_x(matrix_y,matrix_x, matrix_U, n);
	 printf("matrix x is  : \n");
	print_matrix(matrix_x,n,1);
	
	double** matrix_k_pred_transpose = allocate_matrix(1,n);
	transpose(matrix_k_pred, matrix_k_pred_transpose,n,1);
	print_matrix(matrix_k_pred_transpose,1,n);	
	
	double** result = allocate_matrix(1,1);
	multiplyMatrices(matrix_k_pred_transpose,matrix_x,result,1,n,n,1);
	print_matrix(result,1,1);

*/
}
