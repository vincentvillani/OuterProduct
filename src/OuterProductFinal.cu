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

static const int BINSIZE = 1024;

//Not a general outer product kernel, do not use in anything but DSPSR, due to implicit assumptions
//Assumes lhsMatrixLength == rhsMatrixLength
__global__ void wholeOuterProductSum(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, unsigned int resultMatrixGridBlockRowIdx, unsigned int resultMatrixGridBlockColIdx)
{
	//Absolute threadIdx within a block of the results grid
	const int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	//Calculate the index to write the result into the result matrix
	const int indexRowCalc = absoluteThreadIdx / BINSIZE;
	const int indexColCalc = absoluteThreadIdx % BINSIZE;
	const int totalColumnSize = 4 * BINSIZE;

	const int resultRow = (resultMatrixGridBlockRowIdx * BINSIZE) + indexRowCalc;
	const int resultCol = (resultMatrixGridBlockColIdx * BINSIZE) + indexColCalc;
	const int lowerTrianglarLength = (resultRow * (resultRow + 1)) / 2; //calculates the lowerTriangluarLength at this point

	int resultMatrixIdx = (resultRow * totalColumnSize + resultCol) - lowerTrianglarLength;

	//write output to the result matrix
	resultMatrix[resultMatrixIdx] += lhsMatrix[indexRowCalc] * rhsMatrix[indexColCalc];
}


//Assumes square matrices
__device__ __host__ unsigned int upperTriangularLength(unsigned int numRows)
{
	//printf("%d", (numRows * (numRows + 1)) / 2);
	return (numRows * (numRows + 1)) / 2;
}


int main()
{
	upperTriangularLength(2);
	return 0;
}
