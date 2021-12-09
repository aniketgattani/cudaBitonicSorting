/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * This sample implements bitonic sort and odd-even merge sort, algorithms
 * belonging to the class of sorting networks.
 * While generally subefficient on large sequences
 * compared to algorithms with better asymptotic algorithmic complexity
 * (i.e. merge sort or radix sort), may be the algorithms of choice for sorting
 * batches of short- or mid-sized arrays.
 * Refer to the excellent tutorial by H. W. Lang:
 * http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm
 *
 * Victor Podlozhnyuk, 07/09/2009
 */

// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>
#include <helper_timer.h>

#include "sortingNetworks_common.h"
#include <math.h>

using namespace std;

uint *h_InputKey, *h_OutputKeyGPU;
uint *h1_InputKey, *h1_OutputKeyGPU;
uint *d_InputKey,  *d_OutputKey;
double copyTime = 0;

void printArray(uint *a, int size){
    for(int i=0; i < size; i++) printf("%u ", a[i]);
    	printf("\n"); 

}

void copy(uint *dest, uint *src, size_t size, cudaMemcpyKind cudaMemcpyType, StopWatchInterface *timer){
    sdkResetTimer(&timer);
    sdkStartTimer(&timer);
    cudaError_t error = cudaMemcpy(dest, src, size, cudaMemcpyType);
    sdkStopTimer(&timer);
    copyTime += 1.0e-3 * sdkGetTimerValue(&timer);
    checkCudaErrors(error);
}

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    cudaError_t error;
    printf("%s Starting...\n\n", argv[0]);

    printf("Starting up CUDA context...\n");
    int dev = findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *hTimer = NULL;
    StopWatchInterface *hTimerCopy = NULL;

    const uint             N = atoi(argv[1]);
    //const uint          Nmax = 8;
    const uint           DIR = 0;
    const uint     numValues = 65536;
    uint verbose = 0;
    if(argc>=3) atoi(argv[2]);

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
    sdkCreateTimer(&hTimerCopy);

    h_InputKey     = (uint *)malloc(N * sizeof(uint));
    h_OutputKeyGPU = (uint *)malloc(N * sizeof(uint));
    
    srand(2001);

    for (uint i = 0; i < N; i++)
    {
        h_InputKey[i] = rand()%numValues;
    }

    if(verbose) printArray(h_InputKey, N);
    
    printf("Allocating and initializing CUDA arrays...\n\n");
    error = cudaMalloc((void **)&d_InputKey,  min(N, Nmax) * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_OutputKey, min(N, Nmax) * sizeof(uint));
    checkCudaErrors(error);
    
    int flag = 1;

    error = cudaDeviceSynchronize();
    checkCudaErrors(error);

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
            
    uint arrayLength = Nmax;
    uint dir = DIR;
    uint threadCount;

    if(N < Nmax) {
        copy(d_InputKey, h_InputKey, N * sizeof(uint), cudaMemcpyHostToDevice, hTimerCopy);
        
        threadCount = bitonicSort(
            d_OutputKey,
            d_InputKey,
            N,
            0,
            dir
        );

        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

        copy(h_OutputKeyGPU, d_OutputKey, N * sizeof(uint), cudaMemcpyDeviceToHost, hTimerCopy);
    }
    else{
        memcpy(h_OutputKeyGPU, h_InputKey, N * sizeof(uint));
        
    	if(verbose) printArray(h_OutputKeyGPU, N);
        
	   for(arrayLength = Nmax; arrayLength <= N; arrayLength*=2){
            
            for(uint size = arrayLength, stride = arrayLength/2; size >= Nmax; size >>= 1, stride >>= 1){
        	    
                if(verbose) printf("Performing Bitonic Merge using Nmax at a time, size: %u, stride: %u \n", size, stride); 
                
                for(uint i=0; i < N/size; i++){
                    
                    for(uint j=0; j < stride/(Nmax/2); j++){
			             /* dir changes with each even odd subarray of length = arrayLength */
                        dir = ((i*size + j*(Nmax/2))/arrayLength)%2;
                        
                        if(verbose) printf("%u %u %u, %u %u \n", i, j, size, stride, dir);	    
                        
                        h1_OutputKeyGPU = h_OutputKeyGPU + i*size + j*(Nmax/2);
                        
                        copy(d_InputKey, h1_OutputKeyGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice, hTimerCopy);
                        
        	            h1_OutputKeyGPU += stride;
                        d_InputKey += Nmax/2;
                        
                        copy(d_InputKey, h1_OutputKeyGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice, hTimerCopy);
                       
                        d_InputKey -= Nmax/2;
        
                        if(verbose)  printArray(h1_OutputKeyGPU, Nmax/2); 
                                        
        	    	    uint onlyMerge = 1;
                        if(size == Nmax) onlyMerge = 0;
	                    
                        threadCount = bitonicSort(
                            d_OutputKey,
                            d_InputKey,
                            Nmax,	
            			    onlyMerge,
                            dir
                        );
                    
        	            h1_OutputKeyGPU = h_OutputKeyGPU + i*size + j*(Nmax/2);

                        error = cudaDeviceSynchronize();
                        checkCudaErrors(error);

                        copy(h1_OutputKeyGPU, d_OutputKey, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost, hTimerCopy);
                        
			            if(verbose) printArray(h1_OutputKeyGPU, Nmax/2); 
                        
        	            h1_OutputKeyGPU += stride;
                        d_OutputKey += Nmax/2;

                        copy(h1_OutputKeyGPU, d_OutputKey, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost, hTimerCopy);
                        
                        if(verbose) printArray(h1_OutputKeyGPU, Nmax/2); 

                        d_OutputKey -= Nmax/2;
                    }
                }
            }
		}
    }

    uint numIterations = 1;
    sdkStopTimer(&hTimer);
    printf("Average time: %f ms\n\n", sdkGetTimerValue(&hTimer) / numIterations);

    
    double dTimeSecs = 1.0e-3 * sdkGetTimerValue(&hTimer) / numIterations;
    printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, CopyTime = %.5f, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * (double)N/dTimeSecs), dTimeSecs, copyTime, N, 1, threadCount);

    printf("\nValidating the results...\n");
    printf("...reading back GPU results\n");

    int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, 1, N, numValues, DIR, verbose);
    flag = flag && keysFlag;

    printf("\n");

    printf("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    cudaFree(d_OutputKey);
    cudaFree(d_InputKey);
    free(h_OutputKeyGPU);
    free(h_InputKey);

    exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
}
