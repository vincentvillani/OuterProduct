/*
 * HelperFunctions.cpp
 *
 *  Created on: 22/08/2014
 *      Author: vincentvillani
 */

#include "HelperFunctions.h"

//TODO: ADD CEIL CALLS TO BLOCKNUM CALCULATIONS


//vectorLength == binSize of the stokes vectors
//powerOfTwoVectorLength has to be a power of two
//evenResultGridDim has to be an even number
//d_lhsVectorLength == d_rhsVectorLength == powerOfTwoVectorLength
//TODO: TO MANY RESTRICTIONS? WILL PROBABLY STILL WORK IF powerOfTwoVectorLength / evenResultGridDim == an even number
void computeUpperTriangularOuterProduct(float* d_resultMatrix, int resultMatrixLength, float* d_lhsVector,
		float* d_rhsVector, int powerOfTwoVectorLength, int evenResultGridDim, int threadNum)
{
	if(evenResultGridDim % 2 != 0)
	{
		printf("Error: computeUpperTriangularOuterProduct() param 'evenResultGridDim' expects an even number");
		return;
	}

	//calculate number of cuda blocks needed for each kernel
	int cudaWholeOuterProductBlockNum = max((float)1,  min( ceil((powerOfTwoVectorLength * powerOfTwoVectorLength) / threadNum), (float)(1 << 16) - 1));
	int cudaUpperTriOuterProductBlockNum = max((float)1,  min( ceil(((powerOfTwoVectorLength * (powerOfTwoVectorLength + 1) / 2)) / threadNum), (float)(1 << 16) - 1));


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

	/*
	//check for errors
	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
		return;
	}
	*/

	//DEBUG - print results
	//copyAndPrint(d_resultMatrix, resultMatrixLength, evenResultGridDim * powerOfTwoVectorLength);

}


/*
//vectorLength == binSize of the stokes vectors
//powerOfTwoVectorLength has to be a power of two
//evenResultGridDim has to be an even number
//d_lhsVectorLength == d_rhsVectorLength == powerOfTwoVectorLength
//TODO: TO MANY RESTRICTIONS? WILL PROBABLY STILL WORK IF powerOfTwoVectorLength / evenResultGridDim == an even number
void computeUpperTriangularOuterProductStream(float* d_resultMatrix, int resultMatrixLength, float* d_lhsVector,
		float* d_rhsVector, int powerOfTwoVectorLength, int evenResultGridDim, int threadNum, cudaStream_t* stream1,
		cudaStream_t* stream2)
{
	if(evenResultGridDim % 2 != 0)
	{
		printf("Error: computeUpperTriangularOuterProduct() param 'evenResultGridDim' expects an even number");
		return;
	}

	//calculate the number of kernels for each stream
	bool evenStream = false;

	//calculate number of cuda blocks needed for each kernel
	int cudaWholeOuterProductBlockNum = max(1,  min( (powerOfTwoVectorLength * powerOfTwoVectorLength) / threadNum, (1 << 16) - 1));
	int cudaUpperTriOuterProductBlockNum = max(1,  min( ((powerOfTwoVectorLength * (powerOfTwoVectorLength + 1) / 2)) / threadNum, (1 << 16) - 1));


	//for every 'block' in the result matrix
	for(int i = 0; i < evenResultGridDim; ++i)
	{
		if(evenStream)
		{
			//call upper triangular outer product on along the diagonal
			upperTrianglarOuterProductSum<<<cudaUpperTriOuterProductBlockNum, threadNum, 0, *stream1>>>
					(d_resultMatrix, d_lhsVector, d_rhsVector,powerOfTwoVectorLength, evenResultGridDim, i);
		}
		else
		{
			//call upper triangular outer product on along the diagonal
			upperTrianglarOuterProductSum<<<cudaUpperTriOuterProductBlockNum, threadNum, 0, *stream2>>>
					(d_resultMatrix, d_lhsVector, d_rhsVector,powerOfTwoVectorLength, evenResultGridDim, i);
		}

		evenStream = !evenStream; //switch stream

		//call the whole outer product kernel for the remaining blocks on this row
		for(int j = i + 1; j < evenResultGridDim; ++j)
		{
			if(evenStream)
			{
				wholeOuterProductSum<<<cudaWholeOuterProductBlockNum, threadNum, 0, *stream1>>>
						(d_resultMatrix, d_lhsVector, d_rhsVector, powerOfTwoVectorLength, evenResultGridDim, i, j);
			}
			else
			{
				wholeOuterProductSum<<<cudaWholeOuterProductBlockNum, threadNum, 0, *stream2>>>
						(d_resultMatrix, d_lhsVector, d_rhsVector, powerOfTwoVectorLength, evenResultGridDim, i, j);
			}

			evenStream = !evenStream; //switch stream

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
*/



//vectorLength == binSize of the stokes vectors
//powerOfTwoVectorLength has to be a power of two
//evenResultGridDim has to be an even number
//d_lhsVectorLength == d_rhsVectorLength == powerOfTwoVectorLength
//TODO: TO MANY RESTRICTIONS? WILL PROBABLY STILL WORK IF powerOfTwoVectorLength / evenResultGridDim == an even number
void computeUpperTriangularOuterProductOneBigKernel(float*  d_resultMatrix, int resultMatrixLength, float*  d_lhsVector,
		int powerOfTwoVectorLength, int threadNum)
{
	//calculate number of cuda blocks needed for each kernel
	int cudaUpperTriOuterProductBlockNum = max((float)1,  min( ceil(upperTriangularLength(powerOfTwoVectorLength) / threadNum), (float)(1 << 16) - 1));


	//call upper triangular outer product on along the diagonal
	upperTrianglarOuterProductSumOneBigKernel<<<cudaUpperTriOuterProductBlockNum, threadNum>>>
			(d_resultMatrix, d_lhsVector, powerOfTwoVectorLength);



	//check for errors
	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
		return;
	}


	//DEBUG - print results
	copyAndPrint(d_resultMatrix, resultMatrixLength, powerOfTwoVectorLength);


}



void computeOuterProductSmartBruteForce(float* resultMatrix ,float* vec, int vecNCol, int blockDim)
{
	int resultMatrixLength = upperTriangularLength(vecNCol);

	int blockDimX = blockDim;
	int blockDimY = blockDim;
	int gridDimX = ceil((float) vecNCol / blockDimX);
	int gridDimY = ceil((float) vecNCol / blockDimY);
	//int gridDim = ceil((float)vecNCol / blockDim);

	dim3 grid = dim3(gridDimX, gridDimY);
	dim3 block = dim3(blockDimX, blockDimY);

	/*
	printf("Launching kernel with parameters\nGrid(%d, %d), Block(%d, %d)\n"
			"ThreadsPerBlock: %d, TotalThreads: %d\n",
			grid.x, grid.y, block.x, block.y, blockDimX * blockDimY, (blockDimX * blockDimY) * gridDimX * gridDimY);
	*/

	/*
	cudaError_t error = cudaDeviceSynchronize();
	if(error != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error));
		return;
	}
	*/



	outerProductSmartBruteForce<<<grid, block>>>
			(resultMatrix, vec, vecNCol);


	/*
	//check for errors
	//cudaError_t error2 = cudaDeviceSynchronize();
	cudaError_t error2 = cudaPeekAtLastError();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
		return;
	}
	*/

	//DEBUG - print results
	//copyAndPrint(resultMatrix, resultMatrixLength, vecNCol);

}


void computeOuterProductSmartBruteForceLessThreads(float* resultMatrix ,float* vec, int vecNCol, int blockDim)
{
	int resultMatrixLength = upperTriangularLength(vecNCol);

	int blockDimX = blockDim;
	int blockDimY = blockDim;
	int gridDimX = ceil((float) vecNCol / blockDimX);
	int gridDimY = ceil((float) ((vecNCol / 2) + 1) / blockDimY);
	//int gridDim = ceil((float)vecNCol / blockDim);

	dim3 grid = dim3(gridDimX, gridDimY);
	dim3 block = dim3(blockDimX, blockDimY);

	/*
	printf("Launching kernel with parameters\nGrid(%d, %d), Block(%d, %d)\n"
			"ThreadsPerBlock: %d, TotalThreads: %d\n",
			grid.x, grid.y, block.x, block.y, blockDimX * blockDimY, (blockDimX * blockDimY) * gridDimX * gridDimY);
	*/


	//Call kernel
	outerProductSmartBruteForceLessThreads<<<grid, block>>>(resultMatrix, vec, vecNCol);


	/*
	//check for errors
	//cudaError_t error2 = cudaDeviceSynchronize();
	cudaError_t error2 = cudaPeekAtLastError();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
		return;
	}
	*/

	//DEBUG - print results
	//copyAndPrint(resultMatrix, resultMatrixLength, vecNCol);

}


void arbTest(int vectorLength, int resultGridDim)
{
	int resultMatrixLength = upperTriangularLength(vectorLength * resultGridDim);

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



void arbTestOneBigKernel(int vectorLength)
{
	int resultMatrixLength = upperTriangularLength(vectorLength);

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

	computeUpperTriangularOuterProductOneBigKernel(d_resultMatrix, resultMatrixLength, d_vector, vectorLength, 256);

	free(h_vector);

	cudaFree(d_resultMatrix);
	cudaFree(d_vector);

}


void printResultUpperTriangular(float* result, int rowLength, bool genFile)
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
	printResultUpperTriangular(hostData, rowLength, false);
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

		resultMatrixLength = upperTriangularLength(binSize * resultGridDim);
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

			cudaDeviceSynchronize(); //wait till all kernels are finished
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



void runBenchmarkOneBigKernel(int iterations)
{
	float* h_vector;

	float*  d_resultMatrix;
	float*  d_vector;


	clock_t timers[2]; //start and end timers for all 6 bin sizes
	double timingResult; //total elapsed time for each bin size benchmark

	FILE* file = fopen("/mnt/home/vvillani/deviceOuterProductFinal/BenchmarkResults.txt", "w");

	int resultGridDim = 1;
	int binSize;
	int threadSize;
	//const int iterations = 3000;
	int resultMatrixLength;

	fprintf(file, "ITERATIONS: %d\n\n", iterations);

	//for each bin size - 128 to 4096
	for(int i = 0; i < 6; ++i)
	{
		binSize = (1 << (7 + i)) * 4; //4 stokes vectors

		fprintf(file, "\n\n\nBINSIZE: %d\n\n", binSize / 4);

		h_vector = (float*)malloc(sizeof(float) * binSize);

		resultMatrixLength = upperTriangularLength(binSize * resultGridDim);
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
				computeUpperTriangularOuterProductOneBigKernel(d_resultMatrix, resultMatrixLength, d_vector, binSize, threadSize);
			}

			cudaDeviceSynchronize(); //wait till all kernels are finished
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



void runBenchmarkSmartBruteForce(int iterations)
{
	float* h_vector;

	float*  d_resultMatrix;
	float*  d_vector;


	clock_t timers[2]; //start and end timers for all 6 bin sizes
	double timingResult; //total elapsed time for each bin size benchmark

	FILE* file = fopen("/mnt/home/vvillani/deviceOuterProductFinal/BenchmarkResults.txt", "w");

	int resultGridDim = 1;
	int binSize;
	int threadSize;
	//const int iterations = 3000;
	int resultMatrixLength;

	fprintf(file, "ITERATIONS: %d\n\n", iterations);

	//for each bin size - 128 to 4096
	for(int i = 0; i < 6; ++i)
	{
		binSize = (1 << (7 + i)) * 4; //4 stokes vectors

		fprintf(file, "\n\n\nBINSIZE: %d\n\n", binSize / 4);

		h_vector = (float*)malloc(sizeof(float) * binSize);

		resultMatrixLength = upperTriangularLength(binSize * resultGridDim);
		cudaMalloc(&d_resultMatrix, sizeof(float) * resultMatrixLength);
		cudaMalloc(&d_vector, sizeof(float) * binSize);

		cudaMemset(d_resultMatrix, 0, sizeof(float) * resultMatrixLength);

		for(int k = 0; k < binSize; ++k)
			h_vector[k] = k + 1;

		cudaMemcpy(d_vector, h_vector, sizeof(float) * binSize, cudaMemcpyHostToDevice);

		//for each threadSize - 64 to 1024
		for(int j = 0; j < 3; ++j)
		{
			threadSize = 1 << (3 + j);

			setCPUTimer(&timers[0]); //start time

			//perform the benchmark iteration times
			for(int z = 0; z < iterations; ++z)
			{
				computeOuterProductSmartBruteForce(d_resultMatrix, d_vector, binSize, threadSize);
			}

			cudaDeviceSynchronize(); //wait till all kernels are finished
			setCPUTimer(&timers[1]); //end time
			timingResult = calcCPUTime(timers[0], timers[1]); //result

			//write the result to the file
			fprintf(file, "THREADSIZE %d: %f\n", threadSize * threadSize, timingResult);

		}



		free(h_vector);
		cudaFree(d_resultMatrix);
		cudaFree(d_vector);

		printf("Finished iteration %d\n", i);
	}

	fclose(file);
}



void runBenchmarkSmartBruteForceLessThreads(int iterations)
{
	float* h_vector;

	float*  d_resultMatrix;
	float*  d_vector;


	clock_t timers[2]; //start and end timers for all 6 bin sizes
	double timingResult; //total elapsed time for each bin size benchmark

	FILE* file = fopen("/mnt/home/vvillani/deviceOuterProductFinal/BenchmarkResults.txt", "w");

	int resultGridDim = 1;
	int binSize;
	int threadSize;
	//const int iterations = 3000;
	int resultMatrixLength;

	fprintf(file, "ITERATIONS: %d\n\n", iterations);

	//for each bin size - 128 to 4096
	for(int i = 0; i < 6; ++i)
	{
		binSize = (1 << (7 + i)) * 4; //4 stokes vectors

		fprintf(file, "\n\n\nBINSIZE: %d\n\n", binSize / 4);

		h_vector = (float*)malloc(sizeof(float) * binSize);

		resultMatrixLength = upperTriangularLength(binSize * resultGridDim);
		cudaMalloc(&d_resultMatrix, sizeof(float) * resultMatrixLength);
		cudaMalloc(&d_vector, sizeof(float) * binSize);

		cudaMemset(d_resultMatrix, 0, sizeof(float) * resultMatrixLength);

		for(int k = 0; k < binSize; ++k)
			h_vector[k] = k + 1;

		cudaMemcpy(d_vector, h_vector, sizeof(float) * binSize, cudaMemcpyHostToDevice);

		//for each threadSize - 64, 256, 1024
		for(int j = 0; j < 3; ++j)
		{
			threadSize = 1 << (3 + j);

			setCPUTimer(&timers[0]); //start time

			//perform the benchmark iteration times
			for(int z = 0; z < iterations; ++z)
			{
				computeOuterProductSmartBruteForceLessThreads(d_resultMatrix, d_vector, binSize, threadSize);
			}

			cudaDeviceSynchronize(); //wait till all kernels are finished
			setCPUTimer(&timers[1]); //end time
			timingResult = calcCPUTime(timers[0], timers[1]); //result

			//write the result to the file
			fprintf(file, "THREADSIZE %d: %f\n", threadSize * threadSize, timingResult);

		}



		free(h_vector);
		cudaFree(d_resultMatrix);
		cudaFree(d_vector);

		printf("Finished iteration %d\n", i);
	}

	fclose(file);
}



/*
void runBenchmarkStreams(int iterations)
{
	float* h_vector;

	float* d_resultMatrix;
	float* d_vector;

	cudaStream_t stream1;
	cudaStream_t stream2;

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

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

		resultMatrixLength = upperTriangularLength(binSize * resultGridDim);
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
				computeUpperTriangularOuterProductStream(d_resultMatrix, resultMatrixLength, d_vector, d_vector, binSize, resultGridDim, threadSize, &stream1, &stream2);
			}

			cudaDeviceSynchronize(); //wait till all kernels are finished
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
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}

*/

float* computerOuterProductTriangularCPU(float* vec, int vecLength)
{
	float* resultMatrix = (float*)malloc(sizeof(float) * upperTriangularLength(vecLength));

	for(int i = 0; i < upperTriangularLength(vecLength); ++i)
	{
		//Find the corresponding upperTriangluar indices
		int triRowIndex = upperTrianglarRowIndex(i, vecLength);
		int triColumnIndex = upperTriangluarColumnIndexWithRow(i, vecLength, triRowIndex);

		int lowerTrianglarLength = (triRowIndex * (triRowIndex + 1)) / 2; //calculates the lowerTriangluarLength (or offset) at this point
		int resultMatrixIdx = (triRowIndex * vecLength + triColumnIndex) - lowerTrianglarLength;

		resultMatrix[resultMatrixIdx] += vec[triRowIndex] * vec[triColumnIndex];
	}

	return resultMatrix;

}


float* computerOuterProductCPU(float* vec, int vecLength)
{
	int resultLength = vecLength * vecLength;

	//allocate memory for the result
	float* result = (float*)malloc(sizeof(float) * resultLength);

	for(int i = 0; i < resultLength; ++i)
		result[i] = vec[ i / vecLength] * vec[i % vecLength];

	return result;
}


float* extractUpperTriangularElements(float* vec, int vecLength)
{
	int triLength = upperTriangularLength(vecLength);
	float* upperTriElements = (float*)malloc(sizeof(float) * triLength);

	int colLength = vecLength;
	int currentTriIndex = 0;
    int indexOffset = 0;



    //for each row
    for(int i = 0; i < vecLength; ++i)
    {
        //for each column in a row
        for(int j = 0; j < colLength; ++j)
        {
            upperTriElements[currentTriIndex] = vec[i * vecLength + j + indexOffset];
            currentTriIndex++;
        }

        colLength--;
        indexOffset += 1;
    }

	return upperTriElements;

}


int  upperTrianglarRowIndexIntrinsicHost(int idx, int matDim)
{
	int temp = matDim * (matDim + 1) / 2 - 1 - idx;
	int k = floor( (sqrt(8 * temp + 1) - 1) / 2);

	return matDim - 1 - k;
}


void checkCorrectness()
{
	float maxError = 0.005f;

	int vectorSize = 4096 * 4;
	float* h_vector = (float*)malloc(sizeof(float) * vectorSize);
	float* resultBrute;

	for(int i = 0; i < vectorSize; ++i)
		h_vector[i] = i + 1;

	//Do the brute force outer product on the CPU
	resultBrute = computerOuterProductCPU(h_vector, vectorSize);

	float* d_vector;
	float* d_triangularResult;
	float* h_triangularResult;



	int upperTriangularLengthVal = upperTriangularLength(vectorSize);

	h_triangularResult = (float*)malloc(sizeof(float) * upperTriangularLengthVal);

	cudaError_t mallocErr;
	cudaError_t mallocErr2;

	mallocErr = cudaMalloc(&d_vector, sizeof(float) * vectorSize);
	mallocErr2 = cudaMalloc(&d_triangularResult, sizeof(float) * upperTriangularLengthVal);

	cudaMemset(d_triangularResult, 0, sizeof(float) * upperTriangularLengthVal);

	cudaMemcpy(d_vector, h_vector, sizeof(float) * vectorSize, cudaMemcpyHostToDevice);

	if(mallocErr != cudaSuccess || mallocErr2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(mallocErr));
		printf("%s\n", cudaGetErrorString(mallocErr2));
		return;
	}

	// ------------- GPU ALGORITHM HERE --------------------

	//computeOuterProductSmartBruteForce(d_triangularResult, d_vector, vectorSize, 16);
	computeOuterProductSmartBruteForceLessThreads(d_triangularResult, d_vector, vectorSize, 16);

	//computeUpperTriangularOuterProductOneBigKernel
	//(d_triangularResult, upperTriangularLengthVal, d_vector, vectorSize, 256);

	// -----------------------------------------------------

	//check for errors
	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
		return;
	}

	cudaMemcpy(h_triangularResult, d_triangularResult, sizeof(float) * upperTriangularLengthVal, cudaMemcpyDeviceToHost);


	//Are they more or less the same results?


	//get the upper tri results from the brute force result array
	resultBrute = extractUpperTriangularElements(resultBrute, vectorSize);



	for(int i = 0; i < upperTriangularLengthVal; ++i)
	{
		if( abs(resultBrute[i] - h_triangularResult[i]) > maxError)
		{
			printf("Error detected: index: %d, CPU: %f, GPU: %f\n", i, resultBrute[i], h_triangularResult[i]);
		}

	}


	free(h_vector);
	free(resultBrute);
	free(h_triangularResult);

	cudaFree(d_vector);
	cudaFree(d_triangularResult);
}


void testSqrt()
{

	int* h_results;
	int* h_gpuResults;

	int* d_results;

	int nCol = 4096 * 4;
	int num = upperTriangularLength(nCol);


	printf("NUM: %d\n", num);

	h_results = (int*)malloc(sizeof(int) * num);
	h_gpuResults = (int*)malloc(sizeof(int) * num);


	cudaMalloc(&d_results, sizeof(int) * num);
	cudaMemset(d_results, 0, sizeof(int) * num);

	//check for errors
	cudaError_t error2 = cudaDeviceSynchronize();

	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
		return;
	}


	squareRootIntrinsic<<< min((int)ceilf( num / 256), (1 << 16) - 1), 256>>> ( d_results, nCol, num);


	if(error2 != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(error2));
		return;
	}

	for(int i = 0; i < num; ++i)
		h_results[i] = upperTrianglarRowIndexIntrinsicHost(i, nCol);

	cudaMemcpy(h_gpuResults, d_results, sizeof(int) * num, cudaMemcpyDeviceToHost);

	//compare results

	for(int i = 0; i < num; ++i)
	{
		if(h_results[i] != h_gpuResults[i])
		{
			printf("ERORR - Index: %d, CPU: %d, GPU: %d\n", i, h_results[i], h_gpuResults[i]);
		}
	}


	printf("Test complete!\n");

	free(h_results);
	free(h_gpuResults);

	cudaFree(d_results);

}



