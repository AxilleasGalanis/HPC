#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include <omp.h>

#define BLOCK_SIZE 256
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.01 
#define SOFTENING 1e-9f  /* Will guard against denormals */

typedef struct { float4 *pos, *vel; } Body;

void checkCudaErrors(cudaError_t error){
	if(error != cudaSuccess) {
		printf("\033[0;31mCUDA Error: %s in %s, line %d\033[0;37m\n", cudaGetErrorString(error), __FILE__, __LINE__);
	}
}

void randomizeBodies(float *data, int n) {
 srand(100);
 for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(float4 *p, float4 *v, float dt, int n) {
	#pragma omp parallel for
  for (int i = 0; i < n; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz;
  }

}

__global__ void d_bodyForce(float4 *p, float4 *v, float dt, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    
    for (int index = 0; index < gridDim.x; index++) {
      __shared__ float4 shared_pos[BLOCK_SIZE];
      shared_pos[threadIdx.x] = p[index * blockDim.x + threadIdx.x];
      __syncthreads();
	  
      for (int j = 0; j < BLOCK_SIZE; j++) {
        float dx = shared_pos[j].x - p[i].x;
        float dy = shared_pos[j].y - p[i].y;
        float dz = shared_pos[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }
      __syncthreads();
    }

    v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz;
  }
}

int main(const int argc, const char** argv) {

  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = 2*nBodies*sizeof(float4);
  //we save the gpu output to these 
  float *buf = (float*)malloc(bytes); 
  Body p = { (float4*)buf, ((float4*)buf) + nBodies };
  
  //they are used for comparison
  float *h_buf = (float*)malloc(bytes);
  Body h_p = { (float4*)h_buf, ((float4*)h_buf) + nBodies };

  randomizeBodies(buf, 8*nBodies); // Init pos / vel data
  randomizeBodies(h_buf, 8*nBodies); // Init pos / vel data
  
  //GPU initialisation
  float *d_buf;
  cudaMalloc(&d_buf, bytes);
  checkCudaErrors(cudaGetLastError());
  Body d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };
  
  dim3 grid_dim(ceil((double)nBodies / BLOCK_SIZE));

  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    if (iter == 1) {
		bodyForce(h_p.pos, h_p.vel, dt, nBodies); // compute interbody forces
		for (int i = 0 ; i < nBodies; i++) { // integrate position
		   h_p.pos[i].x += h_p.vel[i].x*dt;
      	   h_p.pos[i].y += h_p.vel[i].y*dt;
      	   h_p.pos[i].z += h_p.vel[i].z*dt;
		}
	printf("CPU CALCULATIONS ENDED\n");
	}
    StartTimer();

	cudaMemcpy(d_buf, buf, bytes, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
    d_bodyForce<<<grid_dim, BLOCK_SIZE>>>(d_p.pos, d_p.vel, dt, nBodies); // compute interbody forces
    cudaMemcpy(buf, d_buf, bytes, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaGetLastError());

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p.pos[i].x += p.vel[i].x*dt;
      p.pos[i].y += p.vel[i].y*dt;
      p.pos[i].z += p.vel[i].z*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter == 1) {
		for (int i = 0 ; i < nBodies; i++) {
			if (ABS(h_p.pos[i].x - p.pos[i].x) >= accuracy  || ABS (h_p.pos[i].y - p.pos[i].y) >= accuracy || ABS (h_p.pos[i].z - p.pos[i].z) >= accuracy) {
				printf("ERORR!\n");
				free(buf);
				free(h_buf);
	  			cudaFree(d_buf);
	  			cudaDeviceReset();
				return 0;
			}
		}
	printf("COMPARISONS ENDED SUCCESFULLY\n");
	}
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
  }
  double avgTime = totalTime / (double)(nIters-1); 

  printf("%d Bodies: average %0.3f Billion Interactions / second\nAVERAGE TIME %.3f\n", nBodies, 1e-9 * nBodies * nBodies / avgTime, avgTime);
  free(buf);
  free(h_buf);
  cudaFree(d_buf);
  cudaDeviceReset();
}
