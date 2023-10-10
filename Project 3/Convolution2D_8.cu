/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	5

void checkCudaErrors(cudaError_t error){
	if(error != cudaSuccess) {
		printf("\033[0;31mCUDA Error: %s in %s, line %d\033[0;37m\n", cudaGetErrorString(error), __FILE__, __LINE__);
	}
}



////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = filterR; y < imageH - filterR; y++) {
    for (x = filterR; x < imageW - filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        sum += h_Src[y * imageW + d] * h_Filter[filterR - k];

        h_Dst[y * imageW + x] = sum;
      }
    }
  }

}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;

  for (y = filterR; y < imageH - filterR; y++) {
    for (x = filterR; x < imageW - filterR; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        sum += h_Src[d * imageW + x] * h_Filter[filterR - k];

       	h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}

__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
                       int imageW, int imageH, int filterR) {
  	int k;
                      
	float sum = 0;
	int x = blockIdx.x * blockDim.x + threadIdx.x + filterR;
	int y = blockIdx.y * blockDim.y + threadIdx.y + filterR;
	if(x < imageW && y < imageH){ 
		for (k = -filterR; k <= filterR; k++) {
			int d = x + k;

			sum += d_Src[y  * imageW + d ] * d_Filter[filterR - k];    

		    d_Dst[y * imageW + x] = sum;
	 	}
	}   
}

__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
    			   int imageW, int imageH, int filterR) {

	int k;
  
	float sum = 0;
	int x = blockIdx.x * blockDim.x + threadIdx.x + filterR;
	int y = blockIdx.y * blockDim.y + threadIdx.y + filterR;
	if(x < imageW && y < imageH){ 
		for (k = -filterR; k <= filterR; k++) {
			int d = y + k;

			sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
	 
		    d_Dst[y * imageW + x] = sum;
		}
	}
    
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU,
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;


    int imageW;
    int imageH;
    unsigned int i, j;
	struct timespec tv1, tv2;
	cudaEvent_t start, stop;
	
	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    
    imageW = 2 * filter_radius + imageW;
    imageH = imageW;
    

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)calloc(imageW * imageH , sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
		
	if( !h_Filter || !h_Input|| !h_Buffer || !h_OutputCPU || !h_OutputGPU ){
		printf("Error while allocating host memory.\n");
		exit(1);
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }
	 
    for (i = filter_radius; i < imageW - filter_radius; i++) {
    	for(j = filter_radius; j < imageH - filter_radius; j++){
        	h_Input[i * imageW + j] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    	}  	
    }
    
    printf("Allocating and initializing device arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    cudaMalloc( (void**)&d_Filter, FILTER_LENGTH * sizeof(float));
    //checkCudaErrors(cudaGetLastError());
    cudaMalloc( (void**)&d_Input, imageW * imageH * sizeof(float));
    //checkCudaErrors(cudaGetLastError());
    cudaMalloc( (void**)&d_Buffer, imageW * imageH *sizeof(float));
    //checkCudaErrors(cudaGetLastError());
    cudaMalloc( (void**)&d_OutputGPU,imageW * imageH * sizeof(float));
    //checkCudaErrors(cudaGetLastError());
    
    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaErrors(cudaGetLastError());
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
	
	clock_gettime(CLOCK_MONOTONIC_RAW,&tv1);
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
	clock_gettime(CLOCK_MONOTONIC_RAW,&tv2);
	
	printf("GPU computation...\n");
	
	cudaEventRecord(start);
	
	dim3 block_dim(32, 32);
	dim3 grid_dim(imageW/32, imageH/32);
    
    
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
    
    convolutionRowGPU<<<grid_dim, block_dim>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();
	convolutionColumnGPU<<<grid_dim, block_dim>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
 
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	
	cudaMemcpy( h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(float) , cudaMemcpyDeviceToHost);
	//cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  
    
    cudaEventRecord(stop);
	cudaEventSynchronize(stop);
    float diff = 0;
    cudaEventElapsedTime(&diff, start, stop);

	int flag = 0;
	for (i = filter_radius; i < imageW - filter_radius; i++) {
    	for(j = filter_radius; j < imageH - filter_radius; j++){
        	if( ABS(h_OutputGPU[i * imageW + j] - h_OutputCPU[i * imageW + j]) > accuracy) {
				printf("\033[0;35mGpu output differs\033[0;37m\n");
				flag = 1;
				break;
			 }
		if(flag)
			break;
     	}  	
    }
	
	printf("CPU time = %10f seconds\n", (double) (tv2.tv_nsec - tv1.tv_nsec)/1000000000.0 
	+ (double) (tv2.tv_sec - tv1.tv_sec));
	
	printf("GPU time = %.10f seconds\n", diff / 1000);
	
    // free all the allocated memory
    free(h_OutputCPU);
	free(h_Buffer);
	free(h_Input);
	free(h_Filter);
	free(h_OutputGPU);
	cudaFree(d_OutputGPU);
	cudaFree(d_Buffer);
	cudaFree(d_Input);
	cudaFree(d_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    cudaDeviceReset();


    return 0;
}
