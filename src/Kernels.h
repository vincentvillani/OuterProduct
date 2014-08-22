/*
 * Kernels.h
 *
 *  Created on: 22/08/2014
 *      Author: vincentvillani
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <cuda_runtime.h>
#include <stdio.h>

__device__ __host__ unsigned int upperTriangularLength(unsigned int numRows);

//Converts normal matrix index to an upper trianglar matrix ROW INDEX
__device__ int  upperTrianglarRowIndex(int idx, int matDim);


//Converts normal matrix index to an upper trianglar matrix COLUMN INDEX
__device__ int upperTriangluarColumnIndex(int idx, int matDim);


//Converts normal matrix index to an upper trianglar matrix COLUMN INDEX
__device__ int upperTriangluarColumnIndexWithRow(int idx, int matDim, int rowIdx);


__global__ void wholeOuterProductSum(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, int resultBlockWidth,
		int resultGridWidth, int resultMatrixGridBlockRowIdx, int resultMatrixGridBlockColIdx);


__global__ void upperTrianglarOuterProductSum(float* resultMatrix, float* lhsMatrix, float* rhsMatrix,
		int resultBlockWidth, int resultGridWidth, int resultMatrixGridBlockRowIdx);


__global__ void upperTrianglarOuterProductSumOneBigKernel(float* resultMatrix, float* lhsMatrix, int lhsMatrixLength);

__device__ int  upperTrianglarRowIndexIntrinsic(int idx, int matDim);

__global__ void squareRootIntrinsic(int* results, const int nCol, const int resultSize);

#endif /* KERNELS_H_ */
