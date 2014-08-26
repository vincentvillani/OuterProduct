

#include "Kernels.h"


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


//Assumes square matrices
__device__ __host__ unsigned int upperTriangularLength(unsigned int numRows)
{
	return (numRows * (numRows + 1)) / 2;
}



//Converts normal matrix index to an upper trianglar matrix ROW INDEX
__host__ __device__  int  upperTrianglarRowIndex(int idx, int matDim)
{
	int temp = matDim * (matDim + 1) / 2 - 1 - idx;
	int k = floor( (sqrt((double)8 * temp + 1) - 1) / 2);
	return matDim - 1 - k;
}



//Converts normal matrix index to an upper trianglar matrix COLUMN INDEX
__host__ __device__ int upperTriangluarColumnIndex(int idx, int matDim)
{
	int row = upperTrianglarRowIndex(idx, matDim);
	return idx - matDim * row + row * (row + 1) / 2;
}



//Converts normal matrix index to an upper trianglar matrix COLUMN INDEX
__device__ __host__ int upperTriangluarColumnIndexWithRow(int idx, int matDim, int rowIdx)
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




__global__ void upperTrianglarOuterProductSumOneBigKernel(float* resultMatrix, float* lhsMatrix, int lhsMatrixLength)
{

	int operationsNeeded = (lhsMatrixLength * (lhsMatrixLength + 1)) / 2;

	for(int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x; absoluteThreadIdx < operationsNeeded; absoluteThreadIdx += gridDim.x * blockDim.x)
	{
		//Find the corresponding upperTriangluar indices
		int triRowIndex = upperTrianglarRowIndex(absoluteThreadIdx, lhsMatrixLength);
		int triColumnIndex = upperTriangluarColumnIndexWithRow(absoluteThreadIdx, lhsMatrixLength, triRowIndex);

		int lowerTrianglarLength = (triRowIndex * (triRowIndex + 1)) / 2; //calculates the lowerTriangluarLength (or offset) at this point
		int resultMatrixIdx = (triRowIndex * lhsMatrixLength + triColumnIndex) - lowerTrianglarLength;

		resultMatrix[resultMatrixIdx] += lhsMatrix[triRowIndex] * lhsMatrix[triColumnIndex];

	}
}


//Converts normal matrix index to an upper triangular matrix ROW INDEX
__device__ int  upperTrianglarRowIndexIntrinsic(int idx, int matDim)
{
	int temp = matDim * (matDim + 1) / 2 - 1 - idx;
	int k = floor( ( __dsqrt_rz( (double)8 * temp + 1) - 1) / 2);
	return matDim - 1 - k;
}



__global__ void squareRootIntrinsic(int* results, const int nCol, const int resultSize)
{

	for(int absThreadIdx = blockDim.x * blockIdx.x + threadIdx.x; absThreadIdx < resultSize; absThreadIdx += blockDim.x * gridDim.x)
	{
		results[absThreadIdx] = upperTrianglarRowIndexIntrinsic(absThreadIdx, nCol);
	}
	//printf("%d\n", results[absThreadIdx]);

}


/*
__global__ void outerProductSumBruteForce(float* resultMatrix, float* lhsMatrix, float* rhsMatrix, unsigned int lhsMatrixLength)
{
	//Make each thread do more work if there is any more work to be done
	for(int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x; absoluteThreadIdx < lhsMatrixLength * lhsMatrixLength; absoluteThreadIdx += gridDim.x * blockDim.x)
	{
		//Write back to global memory
		resultMatrix[absoluteThreadIdx] += lhsMatrix[ absoluteThreadIdx / lhsMatrixLength ] * rhsMatrix[absoluteThreadIdx % lhsMatrixLength];
	}
}
*/


/*
__global__ void computeUpperTriangularIndices(int* indices, int nCol, int totalElementsInFullArray)
{
	for(int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x; absoluteThreadIdx < totalElementsInFullArray; absoluteThreadIdx += gridDim.x * blockDim.x)
	{
		//Find the corresponding upperTriangluar indices
		int triRowIndex = upperTrianglarRowIndex(absoluteThreadIdx, nCol);
		int triColumnIndex = upperTriangluarColumnIndexWithRow(absoluteThreadIdx, nCol, triRowIndex);

		int lowerTrianglarLength = (triRowIndex * (triRowIndex + 1)) / 2; //calculates the lowerTriangluarLength (or offset) at this point
		int resultMatrixIdx = (triRowIndex * lhsMatrixLength + triColumnIndex) - lowerTrianglarLength;

		indices[resultMatrixIdx] = resultMatrixIdx;
	}
}
*/


__global__ void outerProductSmartBruteForce(float* resultMatrix, float* vec, int vectorLength)
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x; //column
	int row = (blockIdx.y * blockDim.y) + threadIdx.y; //row


	//check bounds
	if(row >= vectorLength || col >= vectorLength || row > col)
		return;

	int index = (row * vectorLength + col) - (row * (row + 1)) / 2;

	resultMatrix[index] = vec[row] * vec[col];

}




