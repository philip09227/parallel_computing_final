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


struct matrix_node
{
	double x;
	double y;
};

// check for available memory
void checkGpuMem()

{

float free_m,total_m,used_m;

size_t free_t,total_t;

cudaMemGetInfo(&free_t,&total_t);

free_m =(uint)free_t/1048576.0 ;

total_m=(uint)total_t/1048576.0;

used_m=total_m-free_m;

printf ( "  mem free %d .... %f MB mem total %d....%f MB mem used %f MB\n",free_t,free_m,total_t,total_m,used_m);

}

// allocate matrix in host memory 
double** allocate_matrix( int row, int col)
{	
	double **matrix;
	//double *m =(double*)malloc((row+1)*(col+1)*sizeof(double));
	matrix = (double**)malloc( 10000 *sizeof(double*)); 			
	for( int i=0; i<row; i++)
	{
		matrix[i] = (double*)malloc( 10000*sizeof(double));	
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
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			x = (double)((grid[i].x - grid[j].x) * (grid[i].x - grid[j].x))/(2*(l1*l1));
			y = (double)((grid[i].y - grid[j].y) * (grid[i].y - grid[j].y))/(2*(l2*l2));
			matrix_k[i][j] = (1/sqrt((2*3.1415)))*exp(-(x+y));
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
	for(int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{
			if(i==j)
			{	
				identity_matrix[i][j]=1*t;
			}
			else
			{	
			 	identity_matrix[i][j] = 0*t;
			}
		}
	}	
}


void compute_A( double** matrix_A, double** identity_matrix, double** matrix_k, int n)
{
	for( int i=0; i<n; i++)
	{
		for(int j=0; j<n; j++)
		{	
			matrix_A[i][j] = identity_matrix[i][j]+matrix_k[i][j];
		}
	}	
}

// inverse matrix A-1 = adjA/|A|
//adjA = 
//void inverse_matrix()


void LU_Factorization( double ** matrix_A ,double** matrix_U,double** matrix_L, int n)
{
	int i,j,k;
    for(j=0; j<n; j++)
    {
        for(i=0; i<n; i++)
        {
            if(i<=j)
            {
                matrix_U[i][j]=matrix_A[i][j];
                for(k=0; k<i-1; k++)
                   matrix_U[i][j]-=matrix_L[i][k]*matrix_U[k][j];
                if(i==j)
                    matrix_L[i][j]=1;
                else
                    matrix_L[i][j]=0;
            }
            else
            {
                matrix_L[i][j]=matrix_A[i][j];
                for(k=0; k<=j-1; k++)
                    matrix_L[i][j]-=matrix_L[i][k]*matrix_U[k][j];
                matrix_L[i][j]/=matrix_U[j][j];
                matrix_U[i][j]=0;
            }
        }
    }
}


void calculate_y(double ** matrix_f, double ** matrix_y,double ** matrix_L,int n)
{
	int i,j;
    for(i=0; i<n; i++)
    {
        matrix_y[i][0]=matrix_f[i][0];
        for(j=0; j<i; j++)
        {
            matrix_y[i][0]-=matrix_L[i][j]*matrix_y[j][0];
        }
    }
}

void calculate_x(double ** matrix_y, double ** matrix_x,double ** matrix_U,int n)
{
        int i,j;
    for(i=n-1; i>=0; i--)
    {
        matrix_x[i][0]=matrix_y[i][0];
        for(j=i+1; j<n; j++)
        {
            matrix_x[i][0]-=matrix_U[i][j]*matrix_x[j][0];
        }
	matrix_x[i][0]/=matrix_U[i][i];
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
	n = m*m;
	// grid 1/m+1
	matrix_node * grid = (matrix_node * ) malloc((n+1)* sizeof(matrix_node));
	initialize_grid_points(grid,m);
	print_points(grid,m);

	
	
	// observed matrix
	double** matrix = allocate_matrix(n,1);
	assign_observed_value( matrix, grid,m);
	print_matrix(matrix,n,1);
	
	// k n*n matrix
	double** matrix_k = allocate_matrix(n,n);
	construct_matrix_k(matrix_k,grid,n);
	print_matrix(matrix_k,n,n);
	
	// k* n*1 matrix
	double** matrix_k_pred = allocate_matrix(n,1);
	construct_matrix_k(matrix_k_pred,grid,n);
	print_matrix(matrix_k_pred,n,1);
	
		
	// allocat grid in device memory 
	// allocate device memory for grid m*m grid
	matrix_node * device_grid;
        cudaMalloc((void **)&device_grid, (n+1)*sizeof(matrix_node));
	cudaMemcpy(device_grid, grid, (n+1)*sizeof(matrix_node),cudaMemcpyHostToDevice);
        
	
	// allocate matrix in device memory 
	// allocate device memovry for observed points value whis is n*1
	double ** device_observed_points;
	cudaMalloc((void **) device_observed_points, n*sizeof(double *));
	cudaMemcpy(device_observed_points,matrix, n*sizeof(double *),cudaMemcpyHostToDevice);
	checkGpuMem();
	
	//initialise n*n identity matrix
	double ** identity_matrix = allocate_matrix(n,n);
	initialize_identity_matrix(identity_matrix,n);
	print_matrix(identity_matrix,n,n);	
	
	// computer A	
	double ** matrix_A = allocate_matrix(n,n);	
	compute_A( matrix_A, identity_matrix, matrix_k,n);
	print_matrix(matrix_A,n,n);

	double ** matrix_U = allocate_matrix(n,n);
	double ** matrix_L = allocate_matrix(n,n);
	LU_Factorization(matrix_A , matrix_U,matrix_L,n);
	print_matrix(matrix_L,n,n);
	print_matrix(matrix_U,n,n);
		
	double ** matrix_x = allocate_matrix(n,1);
        double ** matrix_y = allocate_matrix(n,1);
	calculate_y(matrix, matrix_y,matrix_L, n);
	print_matrix(matrix_y,n,1);
	calculate_x(matrix_y,matrix_x, matrix_U, n);
	print_matrix(matrix_x,n,1);
	
	double** matrix_k_pred_transpose = allocate_matrix(1,n);
	transpose(matrix_k_pred, matrix_k_pred_transpose,n,1);
	print_matrix(matrix_k_pred_transpose,1,n);	
	
	double** result = allocate_matrix(1,1);
	multiplyMatrices(matrix_k_pred_transpose,matrix_x,result,1,n,n,1);
	print_matrix(result,1,1);


}
