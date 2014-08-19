/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BINSIZE 1024
#define THREADSIZE 256


//Assumes square matrices
__device__ unsigned int upperTriangularLength(unsigned int numRows)
{
	//printf("%d", (numRows * (numRows + 1)) / 2);
	return (numRows * (numRows + 1)) / 2;
}


//Converts normal matrix index to an upper trianglar matrix ROW INDEX
__device__ int  upperTrianglarRowIndex(int idx, int matDim)
{
	int temp = matDim * (matDim + 1) / 2 - 1 - idx;
	int k = floorf( (sqrtf(8 * temp + 1) - 1) / 2);
	return matDim - 1 - k;
}


//Converts normal matrix index to an upper trianglar matrix COLUMN INDEX
__device__ int upperTriangluarColumnIndex(int idx, int matDim)
{
	int row = upperTrianglarRowIndex(idx, matDim);
	return idx - matDim * row + row * (row + 1) / 2;
}


//Converts normal matrix index to an upper trianglar matrix COLUMN INDEX
__device__ int upperTriangluarColumnIndexWithRow(int idx, int matDim, int rowIdx)
{
	return idx - matDim * rowIdx + rowIdx * (rowIdx + 1) / 2;
}



//Not a general outer product kernel, do not use in anything but DSPSR, due to implicit assumptions
__global__ void wholeOuterProductSum(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, int resultMatrixGridBlockRowIdx, int resultMatrixGridBlockColIdx)
{
	//Absolute threadIdx within a block of the results grid
	const int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	//Calculate the index to write the result into the result matrix
	int indexRowCalc = absoluteThreadIdx / BINSIZE;
	int indexColCalc = absoluteThreadIdx % BINSIZE;
	int totalColumnSize = 4 * BINSIZE;

	int resultRow = (resultMatrixGridBlockRowIdx * BINSIZE) + indexRowCalc;
	int resultCol = (resultMatrixGridBlockColIdx * BINSIZE) + indexColCalc;
	int lowerTrianglarLength = (resultRow * (resultRow + 1)) / 2; //calculates the lowerTriangluarLength at this point

	int resultMatrixIdx = (resultRow * totalColumnSize + resultCol) - lowerTrianglarLength;

	//if(threadIdx.x == 0 && blockIdx.x == 1)
	//	printf("ResualtIdx: %d\n", resultMatrixIdx);


	//write output to the result matrix
	resultMatrix[resultMatrixIdx] += lhsMatrix[indexRowCalc] * rhsMatrix[indexColCalc];
}


__global__ void upperTrianglarOuterProduct(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, int resultMatrixGridBlockRowIdx, int resultMatrixGridBlockColIdx)
{
	//Absolute threadIdx within a block of the results grid
	const int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	int triRowIndex = upperTrianglarRowIndex(absoluteThreadIdx, BINSIZE);
	int triColumnIndex = upperTriangluarColumnIndexWithRow(absoluteThreadIdx, BINSIZE, triRowIndex);


	resultMatrix[] += lhsMatrix[triRowIndex] * rhsMatrix[triColumnIndex];


}





/*
row_index(i, M):
    ii = M(M+1)/2-1-i
    K = floor((sqrt(8ii+1)-1)/2)
    return M-1-K

or


unsigned int row_index( unsigned int i, unsigned int M ){
    double m = M;
    double row = (-2*m - 1 + sqrt( (4*m*(m+1) - 8*(double)i - 7) )) / -2;
    if( row == (double)(int) row ) row -= 1;
    return (unsigned int) row;
}


unsigned int column_index( unsigned int i, unsigned int M ){
    unsigned int row = row_index( i, M);
    return  i - M * row + row*(row+1) / 2;
}
*/



void testWholeOuterProduct()
{

	float* d_resultMatrix;
	float* d_lhsMatrix;
	float* d_rhsMatrix;

	cudaMalloc(&d_resultMatrix, sizeof(float) * BINSIZE * BINSIZE * 4 * 4);
	cudaMalloc(&d_lhsMatrix, sizeof(float) * BINSIZE);
	cudaMalloc(&d_rhsMatrix, sizeof(float) * BINSIZE);

	wholeOuterProductSum<<< (BINSIZE * BINSIZE) / THREADSIZE, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 1, 2);

	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
	}

	cudaFree(d_resultMatrix);
	cudaFree(d_lhsMatrix);
	cudaFree(d_rhsMatrix);
}



int main()
{
	//upperTrianglarRowIndex
	//testWholeOuterProduct();
	//upperTriangularLength(2);
	return 0;
}
