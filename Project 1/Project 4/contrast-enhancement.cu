#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

void checkCudaErrors(cudaError_t error){
	if(error != cudaSuccess) {
		printf("\033[0;31mCUDA Error: %s in %s, line %d\033[0;37m\n", cudaGetErrorString(error), __FILE__, __LINE__);
	}
}

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int *hist, *lut;
    cudaEvent_t start, stop;
    unsigned char *source_img, *result_img;
    result.w = img_in.w;
    result.h = img_in.h;

	result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	cudaMalloc((void **)&result_img, img_in.h * img_in.w * sizeof(unsigned char));
    cudaMalloc((void **)&source_img, img_in.h * img_in.w * sizeof(unsigned char));
    cudaMalloc((void **)&hist, 256 * sizeof(int));
    cudaMalloc((void **)&lut, 256 * sizeof(int));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil((double)img_in.w / 16), ceil((double)img_in.h / 16));
    
	//dim3 threads( 16, 4, 1 );
	//int numthreads = threads.x*threads.y;
    //int numblocks = ceil((double)(img_in.w*img_in.h) / (numthreads*255)) ;

    cudaMemset(hist, 0, 256*sizeof(int));
    checkCudaErrors(cudaGetLastError());
    cudaMemcpy(source_img, img_in.img, img_in.h * img_in.w * sizeof(unsigned char), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());

	cudaEventRecord(start);
    d_histogram<<<600, 256>>>(hist, source_img, img_in.h * img_in.w);
    checkCudaErrors(cudaGetLastError());
    d_lut<<<1, 256>>>(lut, hist, img_in.h * img_in.w);
	checkCudaErrors(cudaGetLastError());

    d_histogram_result<<<grid_dim, block_dim>>>(result_img, source_img, lut, img_in.w*img_in.h);
	checkCudaErrors(cudaGetLastError());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float diff = 0;
    cudaEventElapsedTime(&diff, start, stop);
    
    cudaMemcpy(result.img, result_img, img_in.h * img_in.w * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();

    cudaFree(result_img);
    cudaFree(source_img);
    cudaFree(hist);
    cudaFree(lut);
    cudaEventDestroy(start);
	cudaEventDestroy(stop);

    cudaDeviceReset();
	printf("GPU time = %.10f seconds\n", diff / 1000);
    return result;
}
