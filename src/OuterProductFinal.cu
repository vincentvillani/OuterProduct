
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

#include "HelperFunctions.h"
#include "Kernels.h"




int main()
{
	//arbTest(2, 2);

	//runBenchmark(3000);
	//runBenchmarkStreams(3000);

	//runBenchmarkStreams(3000);
	//runBenchmarkOneBigKernel(3000);
	//arbTestOneBigKernel(8);



	//checkCorrectness();

	//runBenchmarkSmartBruteForce(3000);
	runBenchmarkSmartBruteForceLessThreads(3000);

	//cudaDeviceReset();


	return 0;
}
