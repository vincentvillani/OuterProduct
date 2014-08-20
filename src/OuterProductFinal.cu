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
#include <time.h>

//#define MATRIXLENGTH 2 //Matrix width within a 'result matrix grid block'
//#define BINSIZE 6
//#define THREADSIZE 256


// ---------------- DEVICE FUNCTIONS / KERNELS ----------------------------------

//Assumes square matrices
__device__ unsigned int upperTriangularLength(unsigned int numRows)
{
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
void printResultUpperTriangular(T* result, int rowLength, bool genFile)
{
	int numZeros = 0;
	int iterator = 0;

	if(genFile)
	{
		FILE* file = fopen("/mnt/home/vvillani/deviceOuterProductFinal/resultMatrix.txt", "w");

		//for every row
		for(int i = 0; i < rowLength; ++i)
		{
			//print preceding zeros
			for(int j = 0; j < numZeros; ++j)
			{
				fprintf(file, "0, ");
			}

			//print array values
			for(int k = 0; k < rowLength - numZeros; ++k)
			{
				fprintf(file, "%d, ", (int)result[iterator]);
				++iterator;
			}

			fprintf(file, "\n");
			numZeros++;
		}

	}

	numZeros = 0;
	iterator = 0;

	//for every row
	for(int i = 0; i < rowLength; ++i)
	{
		//print preceding zeros
		for(int j = 0; j < numZeros; ++j)
		{
			printf("0, ");
		}

		//print array values
		for(int k = 0; k < rowLength - numZeros; ++k)
		{
			printf("%d, ", (int)result[iterator]);
			++iterator;
		}

		printf("\n");
		numZeros++;
	}

	printf("\n------------------------\n");

}


void copyAndPrint(float* deviceData, int arrayLength, int rowLength)
{
	float* hostData = (float*)malloc(sizeof(float) * arrayLength);
	cudaMemcpy(hostData, deviceData, sizeof(float) * arrayLength, cudaMemcpyDeviceToHost);
	printResultUpperTriangular(hostData, rowLength, true);
}


// ------------------------------------------------------



//Computes the upper triangular outer product of a vector with itself. Because the result matrix will be symmetrical along the
//diagonal, you can save memory by only storing the upper triangular portion of the result and reflecting it along the diagonal
//to obtain the full matrix.

//This function invokes a number of cuda kernel calls


//vectorLength == binSize of the stokes vectors
//powerOfTwoVectorLength has to be a power of two
//evenResultGridDim has to be an even number
//d_lhsVectorLength == d_rhsVectorLength == powerOfTwoVectorLength
//TODO: TO MANY RESTRICTIONS? WILL PROBABLY STILL WORK IF powerOfTwoVectorLength / evenResultGridDim == an even number
void computeUpperTriangularOuterProduct(float* d_resultMatrix, int resultMatrixLength, float* d_lhsVector, float* d_rhsVector, int powerOfTwoVectorLength, int evenResultGridDim, int threadNum)
{
	if(evenResultGridDim % 2 != 0)
	{
		printf("Error: computeUpperTriangularOuterProduct() param 'evenResultGridDim' expects an even number");
		return;
	}

	//calculate number of cuda blocks needed for each kernel
	int cudaWholeOuterProductBlockNum = max(1,  min( (powerOfTwoVectorLength * powerOfTwoVectorLength) / threadNum, (1 << 16) - 1));
	int cudaUpperTriOuterProductBlockNum = max(1,  min( ((powerOfTwoVectorLength * (powerOfTwoVectorLength + 1) / 2)) / threadNum, (1 << 16) - 1));


	//for every 'block' in the result matrix
	for(int i = 0; i < evenResultGridDim; ++i)
	{
		//call upper triangular outer product on along the diagonal
		upperTrianglarOuterProductSum<<<cudaUpperTriOuterProductBlockNum, threadNum>>>
				(d_resultMatrix, d_lhsVector, d_rhsVector,powerOfTwoVectorLength, evenResultGridDim, i);

		//call the whole outer product kernel for the remaining blocks on this row
		for(int j = i + 1; j < evenResultGridDim; ++j)
		{
			wholeOuterProductSum<<<cudaWholeOuterProductBlockNum, threadNum>>>
					(d_resultMatrix, d_lhsVector, d_rhsVector, powerOfTwoVectorLength, evenResultGridDim, i, j);
		}
	}


	//check for errors
	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
		return;
	}

	//DEBUG - print results
	//copyAndPrint(d_resultMatrix, resultMatrixLength, evenResultGridDim * powerOfTwoVectorLength);

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

	computeUpperTriangularOuterProduct(d_resultMatrix, resultMatrixLength, d_vector, d_vector, vectorLength, resultGridDim, 256);

	free(h_vector);

	cudaFree(d_resultMatrix);
	cudaFree(d_vector);

}


void setCPUTimer(clock_t* timer)
{
    *timer = clock();
}


double calcCPUTime(clock_t startTime, clock_t endTime)
{
    return (double)(endTime - startTime) / CLOCKS_PER_SEC;
}



void runBenchmark(int iterations)
{
	float* h_vector;

	float* d_resultMatrix;
	float* d_vector;


	clock_t timers[2]; //start and end timers for all 6 bin sizes
	double timingResult; //total elapsed time for each bin size benchmark

	FILE* file = fopen("/mnt/home/vvillani/deviceOuterProductFinal/BenchmarkResults.txt", "w");

	int resultGridDim = 4;
	int binSize;
	int threadSize;
	//const int iterations = 3000;
	int resultMatrixLength;

	fprintf(file, "ITERATIONS: %d\n\n", iterations);

	//for each bin size - 128 to 4096
	for(int i = 0; i < 6; ++i)
	{
		binSize = 1 << (7 + i);

		fprintf(file, "\n\n\nBINSIZE: %d\n\n", binSize);

		h_vector = (float*)malloc(sizeof(float) * binSize);

		resultMatrixLength = upperTriLength(binSize * resultGridDim);
		cudaMalloc(&d_resultMatrix, sizeof(float) * resultMatrixLength);
		cudaMalloc(&d_vector, sizeof(float) * binSize);

		cudaMemset(d_resultMatrix, 0, sizeof(float) * resultMatrixLength);

		for(int k = 0; k < binSize; ++k)
			h_vector[k] = k + 1;

		cudaMemcpy(d_vector, h_vector, sizeof(float) * binSize, cudaMemcpyHostToDevice);

		//for each threadSize - 64 to 1024
		for(int j = 0; j < 5; ++j)
		{
			threadSize = 1 << (6 + j);

			setCPUTimer(&timers[0]); //start time

			//perform the benchmark iteration times
			for(int z = 0; z < iterations; ++z)
			{
				computeUpperTriangularOuterProduct(d_resultMatrix, resultMatrixLength, d_vector, d_vector, binSize, resultGridDim, threadSize);
			}

			setCPUTimer(&timers[1]); //end time
			timingResult = calcCPUTime(timers[0], timers[1]); //result

			//write the result to the file
			fprintf(file, "THREADSIZE %d: %f\n", threadSize, timingResult);

		}



		free(h_vector);
		cudaFree(d_resultMatrix);
		cudaFree(d_vector);

		printf("Finished iteration %d\n", i);
	}

	fclose(file);
}


int main()
{
	//arbTest(2, 2);
	//arbTest(4, 2);
	//arbTest(4, 4);

	//arbTest(512, 4);

	runBenchmark(3000);

	//arbTest(1024, 4);

	return 0;
}
