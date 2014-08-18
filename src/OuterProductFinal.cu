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

static const int WORK_SIZE = 256;

//Assumes lhsMatrixLength == rhsMatrixLength
__global__ void outerProductSum(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, unsigned int lhsMatrixLength)
{
	//Make each thread do more work if there is any more work to be done
	for(int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x; absoluteThreadIdx < lhsMatrixLength * lhsMatrixLength; absoluteThreadIdx += gridDim.x * blockDim.x)
	{
		//Write back to global memory
		resultMatrix[absoluteThreadIdx] += lhsMatrix[ (int)floorf(absoluteThreadIdx / lhsMatrixLength) ] * rhsMatrix[absoluteThreadIdx % lhsMatrixLength];
	}
}

int main()
{
	return 0;
}
