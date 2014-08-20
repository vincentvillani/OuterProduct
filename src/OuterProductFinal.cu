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
#include <iostream>
//#define MATRIXLENGTH 2 //Matrix width within a 'result matrix grid block'
//#define BINSIZE 6
#define THREADSIZE 256

//TODO: remove either resultMatrixGridBlockRowIdx or resultMatrixGridBlockColIdx from the upperTriangular kernel as they will always be the same


// ---------------- DEVICE FUNCTIONS / KERNELS ----------------------------------

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
__global__ void wholeOuterProductSum(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, int resultBlockWidth, int resultGridWidth, int resultMatrixGridBlockRowIdx, int resultMatrixGridBlockColIdx)
{

	//Absolute threadIdx within a block of the results grid
	//const int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

	int threadsNeeded = resultBlockWidth * resultBlockWidth;

	for(int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x; absoluteThreadIdx < threadsNeeded; absoluteThreadIdx += gridDim.x * blockDim.x)
	{

		//Calculate the index to write the result into the result matrix
		int localIndexRowCalc = absoluteThreadIdx / resultBlockWidth;
		int localIndexColCalc = absoluteThreadIdx % resultBlockWidth;

		int totalColumnSize = resultBlockWidth * resultGridWidth;

		int resultRow = (resultMatrixGridBlockRowIdx * resultBlockWidth) + localIndexRowCalc;
		int resultCol = (resultMatrixGridBlockColIdx * resultBlockWidth) + localIndexColCalc;
		int lowerTrianglarLength = (resultRow * (resultRow + 1)) / 2; //calculates the lowerTriangluarLength (or offset) at this point

		int resultMatrixIdx = (resultRow * totalColumnSize + resultCol) - lowerTrianglarLength;

		/*
		if(absoluteThreadIdx == 0)
		{
			printf("resultRow: %d\n", resultRow);
			printf("localIndexColCalc: %d\n", localIndexColCalc);
			printf("resultCol: %d\n", resultCol);
			printf("resultMatrixIdx: %d\n", resultMatrixIdx);
		}
		*/

		//Calculate the lhsMatrix and rhsMatrix indices to get elements from
		//int lhsMatIndex = localIndexRowCalc + (resultMatrixGridBlockRowIdx * resultBlockWidth);
		//int rhsMatIndex = localIndexColCalc + (resultMatrixGridBlockColIdx * resultBlockWidth);

		//write output to the result matrix
		resultMatrix[resultMatrixIdx] += lhsMatrix[localIndexRowCalc] * rhsMatrix[localIndexColCalc];
	}
}

//resultMatrixGridBlockRowIdx == resultMatrixGridBlockColIdx, due to this operating along the diagonal of a matrix
__global__ void upperTrianglarOuterProductSum(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, int resultBlockWidth, int resultGridWidth, int resultMatrixGridBlockRowIdx)
{

	int threadsNeeded = (resultBlockWidth * (resultBlockWidth + 1)) / 2;

	for(int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x; absoluteThreadIdx < threadsNeeded; absoluteThreadIdx += gridDim.x * blockDim.x)
	{

		//Find the corresponding upperTriangluar indices
		int triRowIndex = upperTrianglarRowIndex(absoluteThreadIdx, resultBlockWidth);
		int triColumnIndex = upperTriangluarColumnIndexWithRow(absoluteThreadIdx, resultBlockWidth, triRowIndex);

		int totalColumnSize =  resultBlockWidth * resultGridWidth;

		//Compute the position in the resultMatrix to store the result
		int resultRow = (resultMatrixGridBlockRowIdx * resultBlockWidth) + triRowIndex;
		int resultCol = (resultMatrixGridBlockRowIdx * resultBlockWidth) + triColumnIndex;

		int lowerTrianglarLength = (resultRow * (resultRow + 1)) / 2; //calculates the lowerTriangluarLength (or offset) at this point
		int resultMatrixIdx = (resultRow * totalColumnSize + resultCol) - lowerTrianglarLength;


		//Calculate the lhsMatrix and rhsMatrix indices to get elements from
		//int lhsMatIndex = triRowIndex + (resultMatrixGridBlockRowIdx * resultBlockWidth);
		//int rhsMatIndex = triColumnIndex + (resultMatrixGridBlockColIdx * resultBlockWidth);

		/*
		if(absoluteThreadIdx == 2)
		{
			printf("triRow: %d\n", triRowIndex);
			printf("triCol: %d\n", resultCol);
			printf("resultMatrixIdx: %d\n", resultMatrixIdx);
			printf("lhsMatIndex: %d\n", lhsMatIndex);
			printf("rhsMatIndex: %d\n", rhsMatIndex);
		}
		*/

		resultMatrix[resultMatrixIdx] += lhsMatrix[triRowIndex] * rhsMatrix[triColumnIndex];
	}
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

// ------------------------------------------------------------


// -------------- DEBUG ---------------------------------------

template<typename T>
void printResultUpperTriangular(T* result, int rowLength)
{
	int numZeros = 0;
	int iterator = 0;

	//for every row
	for(int i = 0; i < rowLength; ++i)
	{
		//print preceding zeros
		for(int j = 0; j < numZeros; ++j)
		{
			std::cout << "0, ";
		}

		//print array values
		for(int k = 0; k < rowLength - numZeros; ++k)
		{
			std::cout << result[iterator] << ", ";
			++iterator;
		}

		std::cout << std::endl;
		numZeros++;
	}

	std::cout << "\n------------------------\n" << std::endl;

}


void copyAndPrint(float* deviceData, int arrayLength, int rowLength)
{
	float* hostData = (float*)malloc(sizeof(float) * arrayLength);
	cudaMemcpy(hostData, deviceData, sizeof(float) * arrayLength, cudaMemcpyDeviceToHost);
	printResultUpperTriangular(hostData, rowLength);
}




//Makes a 4x4 result matrix out of two 4 element vectors
void testOuterProductRoutine4x4()
{
	float* h_lhsMatrix;
	float* h_rhsMatrix;

	float* d_resultMatrix;
	float* d_lhsMatrix;
	float* d_rhsMatrix;

	int resultLength = (4 * (4 + 1)) / 2;

	h_lhsMatrix = (float*)malloc(sizeof(float) * 2);
	h_rhsMatrix = (float*)malloc(sizeof(float) * 2);

	cudaMalloc(&d_resultMatrix, sizeof(float) * resultLength);
	cudaMalloc(&d_lhsMatrix, sizeof(float) * 2);
	cudaMalloc(&d_rhsMatrix, sizeof(float) * 2);

	cudaMemset(d_resultMatrix, 0, sizeof(float) * resultLength);


	for(int i = 0; i < 2; ++i)
	{
		h_lhsMatrix[i] = i + 1;
		h_rhsMatrix[i] = i + 1;
	}

	cudaMemcpy(d_lhsMatrix, h_lhsMatrix, sizeof(float) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rhsMatrix, h_rhsMatrix, sizeof(float) * 2, cudaMemcpyHostToDevice);

	//make the kernel calls to compute the result matrix

	//top left hand corner
	upperTrianglarOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 2, 2, 0);
	wholeOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 2, 2, 0, 1);
	upperTrianglarOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 2, 2, 1);


	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
	}

	copyAndPrint(d_resultMatrix, resultLength, 4);



	cudaFree(d_resultMatrix);
	cudaFree(d_lhsMatrix);
	cudaFree(d_rhsMatrix);
}


//Makes a 4x4 result matrix out of two 4 element vectors
void testOuterProductRoutine6x6()
{
	float* h_lhsMatrix;
	float* h_rhsMatrix;

	float* d_resultMatrix;
	float* d_lhsMatrix;
	float* d_rhsMatrix;

	h_lhsMatrix = (float*)malloc(sizeof(float) * 6);
	h_rhsMatrix = (float*)malloc(sizeof(float) * 6);

	int resultLength = (6 * (6 + 1)) / 2;

	cudaMalloc(&d_resultMatrix, sizeof(float) * resultLength);
	cudaMalloc(&d_lhsMatrix, sizeof(float) * 6);
	cudaMalloc(&d_rhsMatrix, sizeof(float) * 6);

	cudaMemset(d_resultMatrix, 0, sizeof(float) * resultLength);


	for(int i = 0; i < 6; ++i)
	{
		h_lhsMatrix[i] = i + 1;
		h_rhsMatrix[i] = i + 1;
	}

	cudaMemcpy(d_lhsMatrix, h_lhsMatrix, sizeof(float) * 6, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rhsMatrix, h_rhsMatrix, sizeof(float) * 6, cudaMemcpyHostToDevice);

	//make the kernel calls to compute the result matrix

	//top left hand corner
	upperTrianglarOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 2, 3, 0);

	//top middle
	wholeOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 2, 3, 0, 1);

	//top right
	wholeOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 2, 3, 0, 2);

	//middle
	upperTrianglarOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 2, 3, 1);

	//middle right
	wholeOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 2, 3, 1, 2);

	//bottom right
	upperTrianglarOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 2, 3, 2);

	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
	}

	copyAndPrint(d_resultMatrix, resultLength, 6);



	cudaFree(d_resultMatrix);
	cudaFree(d_lhsMatrix);
	cudaFree(d_rhsMatrix);
}



void testOuterProductRoutine9x9()
{
	int resultGridDim = 9; //assumes its square
	//int gridDim = 3;

	float* h_lhsMatrix;
	float* h_rhsMatrix;

	float* d_resultMatrix;
	float* d_lhsMatrix;
	float* d_rhsMatrix;

	h_lhsMatrix = (float*)malloc(sizeof(float) * resultGridDim);
	h_rhsMatrix = (float*)malloc(sizeof(float) * resultGridDim);

	int resultLength = (resultGridDim * (resultGridDim + 1)) / 2;

	cudaMalloc(&d_resultMatrix, sizeof(float) * resultLength);
	cudaMalloc(&d_lhsMatrix, sizeof(float) * resultGridDim);
	cudaMalloc(&d_rhsMatrix, sizeof(float) * resultGridDim);

	cudaMemset(d_resultMatrix, 0, sizeof(float) * resultLength);


	for(int i = 0; i < resultGridDim; ++i)
	{
		h_lhsMatrix[i] = i + 1;
		h_rhsMatrix[i] = i + 1;
	}

	cudaMemcpy(d_lhsMatrix, h_lhsMatrix, sizeof(float) * resultGridDim, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rhsMatrix, h_rhsMatrix, sizeof(float) * resultGridDim, cudaMemcpyHostToDevice);

	//make the kernel calls to compute the result matrix

	//Top left
	upperTrianglarOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 3, 3, 0);

	//Top center
	wholeOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 3, 3, 0, 1);

	//Top right
	wholeOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 3, 3, 0, 2);

	//Middle Center
	upperTrianglarOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 3, 3, 1);

	//Middle Right
	wholeOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 3, 3, 1, 2);

	//Bottom right
	upperTrianglarOuterProductSum<<<1, THREADSIZE>>>(d_resultMatrix, d_lhsMatrix, d_rhsMatrix, 3, 3, 2);


	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
	}

	copyAndPrint(d_resultMatrix, resultLength, resultGridDim);



	cudaFree(d_resultMatrix);
	cudaFree(d_lhsMatrix);
	cudaFree(d_rhsMatrix);
}



// ------------------------------------------------------



//Computes the upper triangular outer product of a vector with itself. Because the result matrix will be symmetrical along the
//diagonal, you can save memory by only storing the upper triangular portion of the result and reflecting it along the diagonal
//to obtain the full matrix.

//This function invokes a number of cuda kernel calls


//vectorLength == binSize of the stokes vectors
void computeUpperTriangularOuterProduct(float* d_resultMatrix, int resultMatrixLength, float* d_vector, int vectorLength, int resultGridDim)
{

	//calculate number of cuda blocks needed for each kernel
	int cudaWholeOuterProductBlockNum = max(1,  min( (vectorLength * vectorLength) / THREADSIZE, (1 << 16) - 1));
	int cudaUpperTriOuterProductBlockNum = max(1,  min( ((vectorLength * (vectorLength + 1) / 2)) / THREADSIZE, (1 << 16) - 1));


	//for every 'block' in the result matrix
	for(int i = 0; i < resultGridDim; ++i)
	{
		//call upper triangular outer product on along the diagonal
		upperTrianglarOuterProductSum<<<cudaUpperTriOuterProductBlockNum, THREADSIZE>>>
				(d_resultMatrix, d_vector, d_vector, vectorLength, resultGridDim, i);

		//call the whole outer product kernel for the remaining blocks on this row
		for(int j = i + 1; j < resultGridDim; ++j)
		{
			wholeOuterProductSum<<<cudaWholeOuterProductBlockNum, THREADSIZE>>>
					(d_resultMatrix, d_vector, d_vector, vectorLength, resultGridDim, i, j);
		}
	}

	//TODO: REMOVE THIS
	//wholeOuterProductSum<<<cudaWholeOuterProductBlockNum, THREADSIZE>>>
	//					(d_resultMatrix, d_vector, d_vector, vectorLength, resultGridDim, 0, 1);


	//check for errors
	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
		return;
	}

	//DEBUG - print results
	copyAndPrint(d_resultMatrix, resultMatrixLength, resultGridDim * vectorLength);

}

int upperTriLength(int rowSize)
{
	return (rowSize * (rowSize + 1)) / 2;
}


void arbTest(int vectorLength, int resultGridDim)
{
	int resultMatrixLength = upperTriLength(vectorLength * resultGridDim);

	float* h_vector;

	float* d_resultMatrix;
	float* d_vector;

	h_vector = (float*)malloc(sizeof(float) * vectorLength);


	cudaMalloc(&d_resultMatrix, sizeof(float) * resultMatrixLength);
	cudaMalloc(&d_vector, sizeof(float) * vectorLength);

	cudaMemset(d_resultMatrix, 0, sizeof(float) * resultMatrixLength);


	for(int i = 0; i < vectorLength; ++i)
	{
		h_vector[i] = i + 1;
	}

	cudaMemcpy(d_vector, h_vector, sizeof(float) * vectorLength, cudaMemcpyHostToDevice);

	computeUpperTriangularOuterProduct(d_resultMatrix, resultMatrixLength, d_vector, vectorLength, resultGridDim);

}


int main()
{
	//testOuterProductRoutine4x4();
	//testOuterProductRoutine6x6();
	//testOuterProductRoutine9x9();
	//testOuterProductRoutine512x512();

	arbTest(2, 2);
	arbTest(4, 2);
	arbTest(4, 4);

	return 0;
}
