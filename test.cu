
#include<stdio.h>

#include <stdlib.h>
#include <math.h>
#define MAX 100
using namespace std;

 float det(float a[MAX][MAX], int n);
 void minor(float b[MAX][MAX], float a[MAX][MAX], int i, int n);

void minor(float b[MAX][MAX], float a[MAX][MAX], int i, int n)
	int j, l, h = 0, k = 0;
	for (l = 1; l < n; l++)
		for (j = 0; j < n; j++) {//此处不用考虑会不会取到第i行，因为l=i+1，自然地避开了取第i行的风险
			if (j == i)//假设进行到第i列
				continue;//不加入余子式中
			b[h][k] = a[l][j];//如果不是第i列，余子式先从第一行开始填满
			k++;			  //如在求a[0][0]的余子式，则不考虑第0列的
			if (k == (n - 1)) {//当第一列填满时，h++，进行下一行的填充
				h++;
				k = 0;//k回到第一列，准备继续向后填充
			}
		}
}// end function

//---------------------------------------------------solved
//	calculate determinte of matrix//先计算行列式，判断矩阵能否求逆矩阵
float det(float a[MAX][MAX], int n) {
	int i;
	float b[MAX][MAX], sum = 0;
	if (n == 1)//如果行列式为一阶行列式
		return a[0][0];
	else if (n == 2)//如果为二阶行列式
		return (a[0][0] * a[1][1] - a[0][1] * a[1][0]);
	else
		for (i = 0; i < n; i++) {
			minor(b, a, i, n);	// read function//这样可以求得行列式的余子式
			sum = (float)(sum + a[0][i] * pow(-1, i)*det(b, (n - 1)));	// read function	// sum = determinte matrix
			//在后面出现递归，依次降维矩阵的维数，让矩阵满足n==2的条件即可计算出矩阵的行列式
		}
	return sum;
}// end function

int main(void) {
	int n;
	float a[MAX][MAX], d[MAX][MAX], deter;
	cout << "\n C++ Program To Find Inverse Of Matrix\n\n";
	n = scanf(a);	//task1:首先输入预想要进行求逆的矩阵
	int print_matrix = 1;
	printf(a, n, print_matrix);	//将输入的代码打印出来
	deter = (float)det(a, n);	//task2:求行列式，判断矩阵能否求逆
}
