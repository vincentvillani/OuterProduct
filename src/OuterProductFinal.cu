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

	testSqrt();

	//cudaDeviceReset();


	return 0;
}
