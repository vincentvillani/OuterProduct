
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

#include "HelperFunctions.h"
#include "Kernels.h"

//NEED TO USE THE DOUBLE PRECISION SQRT :(
//TODO: benchmarks
//TODO: Check that a brute force outer product kernel produces the same results in the upper product part of the matrix as
//the upper product kernel


int main()
{
	//arbTest(2, 2);

	//runBenchmark(3000);
	//runBenchmarkStreams(3000);

	//runBenchmarkStreams(3000);
	//runBenchmarkOneBigKernel(3000);
	//arbTestOneBigKernel(8);

	//runBenchmarkSmartBruteForce(3000);
	//testSqrt();
	checkCorrectness();

	//cudaDeviceReset();


	return 0;
}
