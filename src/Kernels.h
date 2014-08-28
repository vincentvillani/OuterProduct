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
#include <cuComplex.h>

__device__ __host__ unsigned int upperTriangularLength(unsigned int numRows);

//Converts normal matrix index to an upper trianglar matrix ROW INDEX
__device__ __host__ int  upperTrianglarRowIndex(int idx, int matDim);


//Converts normal matrix index to an upper trianglar matrix COLUMN INDEX
__device__ __host__ int upperTriangluarColumnIndex(int idx, int matDim);


//Converts normal matrix index to an upper trianglar matrix COLUMN INDEX
__device__ __host__ int upperTriangluarColumnIndexWithRow(int idx, int matDim, int rowIdx);


__global__ void wholeOuterProductSum(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, int resultBlockWidth,
		int resultGridWidth, int resultMatrixGridBlockRowIdx, int resultMatrixGridBlockColIdx);


__global__ void upperTrianglarOuterProductSum(float* resultMatrix, float* lhsMatrix, float* rhsMatrix,
		int resultBlockWidth, int resultGridWidth, int resultMatrixGridBlockRowIdx);


__global__ void upperTrianglarOuterProductSumOneBigKernel(float* resultMatrix, float* lhsMatrix, int lhsMatrixLength);

__device__ int  upperTrianglarRowIndexIntrinsic(int idx, int matDim);

__global__ void squareRootIntrinsic(int* results, const int nCol, const int resultSize);

//__global__ void outerProductSumBruteForce(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, unsigned int lhsMatrixLength);

//__global__ void computeUpperTriangularIndices(int* resultMatrix, int nCol, int numberOfElements);



__global__ void outerProductSmartBruteForce(float* resultMatrix, float* vec, int vectorLength);

__global__ void outerProductSmartBruteForceLessThreads(float* resultMatrix, float* vec, int vectorLength);

//Specialised outer product for DSPSR
__global__ void outerProductUpperTri(cuFloatComplex* resultMatrix, cuFloatComplex* vec, unsigned int vectorLength);


__global__ void normalise(float* result, unsigned int resultLength, float* amps, unsigned int* hits);

#endif /* KERNELS_H_ */
