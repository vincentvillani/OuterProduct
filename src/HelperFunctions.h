
#ifndef HELPER_FUNCTION_H
#define HELPER_FUNCTION_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "Kernels.h"
#include <cmath>
#include <time.h>



void computeUpperTriangularOuterProduct(float* d_resultMatrix, int resultMatrixLength, float* d_lhsVector,
		float* d_rhsVector, int powerOfTwoVectorLength, int evenResultGridDim, int threadNum);



/*void computeUpperTriangularOuterProductStream(float* d_resultMatrix, int resultMatrixLength, float* d_lhsVector,
		float* d_rhsVector, int powerOfTwoVectorLength, int evenResultGridDim, int threadNum, cudaStream_t* stream1,
		cudaStream_t* stream2);
		*/


void computeUpperTriangularOuterProductOneBigKernel(float*  d_resultMatrix, int resultMatrixLength, float*  d_lhsVector,
		int powerOfTwoVectorLength, int threadNum);

void computeOuterProductSmartBruteForce(float* resultMatrix ,float* vec, int vecNCol, int blockDim);

void computeOuterProductSmartBruteForceLessThreads(float* resultMatrix ,float* vec, int vecNCol, int blockDim);

float* computerOuterProductCPU(float* vec, int vecLength);
float* computeOuterProductUpperTriCPU(float* vec, int vecLength);

void arbTest(int vectorLength, int resultGridDim);


void arbTestOneBigKernel(int vectorLength);


void printResultUpperTriangular(float* result, int rowLength, bool genFile);
void copyAndPrint(float* deviceData, int arrayLength, int rowLength);

//Benchmarking timing
void setCPUTimer(clock_t* timer);
double calcCPUTime(clock_t startTime, clock_t endTime);

void runBenchmark(int iterations);
void runBenchmarkOneBigKernel(int iterations);
void runBenchmarkStreams(int iterations);
void runBenchmarkSmartBruteForce(int iterations);
void runBenchmarkSmartBruteForceLessThreads(int iterations);


int  upperTrianglarRowIndexIntrinsicHost(int idx, int matDim);
void testSqrt();
void checkCorrectness();
void checkCorrectnessCPU();



#endif
