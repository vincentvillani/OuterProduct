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
#define THREADSIZE 1024

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

	//Calculate the index to write the result into the result matrix
	int indexRowCalc = absoluteThreadIdx / BINSIZE;
	int indexColCalc = absoluteThreadIdx % BINSIZE;
	int totalColumnSize = 4 * BINSIZE;

	int offsetCalc = (indexRowCalc * (indexRowCalc + 1) / 2);



	//LHS is correct, RHS is not correct yet
	//Result matrix has not been attempted yet

	resultMatrix[] += lhsMatrix[(absoluteThreadIdx + offsetCalc) / BINSIZE] *
			rhsMatrix[];
}


//Assumes square matrices
__device__ __host__ unsigned int upperTriangularLength(unsigned int numRows)
{
	//printf("%d", (numRows * (numRows + 1)) / 2);
	return (numRows * (numRows + 1)) / 2;
}


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
	testWholeOuterProduct();
	//upperTriangularLength(2);
	return 0;
}
