#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"




__global__ void d_histogram(int *hist_out, unsigned char *img_in, int img_size){

 	__shared__ int sHist[256];
 	for ( int i = threadIdx.x; i < 256; i += blockDim.x ) {
        sHist[i] = 0;
    }

    __syncthreads();

    for ( int i = blockIdx.x*blockDim.x+threadIdx.x; i < img_size/4; i += blockDim.x*gridDim.x ) {
		unsigned int val = ((unsigned int*)img_in)[i];
		unsigned int index = 0xff&((val & 0xff)+blockIdx.x) ;
    	atomicAdd( &sHist[index], 1 ); val >>=8;

    	index = 0xff&((val & 0xff)+blockIdx.x) ;
    	atomicAdd( &sHist[index], 1 ); val >>=8;

    	index = 0xff&((val & 0xff)+blockIdx.x) ;
    	atomicAdd( &sHist[index], 1 ); val >>=8;

    	index = 0xff&((val & 0xff)+blockIdx.x) ;
    	atomicAdd( &sHist[index], 1 );
    }

    __syncthreads();
    for ( int i = threadIdx.x; i < 256; i += blockDim.x) {
    	unsigned index = (0xff&(i+blockIdx.x));
        atomicAdd( &hist_out[i], sHist[index]);
    }
}


__global__ void d_lut(int *lut, int *hist, int img_size){
	__shared__ int temp[2*256];
	temp[threadIdx.x] = hist[threadIdx.x];
	temp[threadIdx.x + 256] = 0;
        __shared__ int min;
	int i = 0;
	if(threadIdx.x == 0){
		min = 0;
		while(min == 0){
			min = hist[i++];
		}
	}

	__syncthreads();
	int d = img_size  - min;
	for(unsigned int stride = 1;stride <= 256; stride *= 2){
		int id = (threadIdx.x+1) * stride * 2 - 1;
		if(id < 2*256)
			temp[id] += temp[id - stride];
		__syncthreads();
	}

	for(unsigned int stride = 256/2; stride > 0; stride /= 2){
		__syncthreads();
		int id = (threadIdx.x+1) * stride * 2 - 1;
		if(id+stride < 2*256)
			temp[id + stride] += temp[id];
	}

	__syncthreads();

	lut[threadIdx.x] = (int)(((float)temp[threadIdx.x] - min)*255/d + 0.5); 
	if (lut[threadIdx.x] < 0)
		lut[threadIdx.x] = 0;

}

__global__ void d_histogram_result(unsigned char * img_out, unsigned char * img_in, 
                            int * lut, int img_size){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = x + (blockDim.x*gridDim.x) *  y;

	if(index<img_size){
		img_out[index] = (unsigned char)lut[img_in[index]];
        }
}
