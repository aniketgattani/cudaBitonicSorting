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


uint *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;
uint *h1_InputKey, *h1_InputVal, *h1_OutputKeyGPU, *h1_OutputValGPU;
uint *d_InputKey, *d_InputVal,    *d_OutputKey,    *d_OutputVal;

void resetPointers(){
    h1_InputKey = h_InputKey;
    h1_InputVal = h_InputVal;
    h1_OutputKeyGPU = h_OutputKeyGPU;
    h1_OutputValGPU = h_OutputValGPU;
}

void printArray(uint *a, int size){
    for(int i=0; i < size; i++) printf("%u ", a[i]);
    printf("\n"); 
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

    const uint             N = atoi(argv[1]);
    const uint          Nmax = 8;
    const uint           DIR = 0;
    const uint     numValues = 65536;

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
    h_InputKey     = (uint *)malloc(N * sizeof(uint));
    h_InputVal     = (uint *)malloc(N * sizeof(uint));
    h_OutputKeyGPU = (uint *)malloc(N * sizeof(uint));
    h_OutputValGPU = (uint *)malloc(N * sizeof(uint));
    temp = (uint *)malloc(N * sizeof(uint));
    srand(2001);

    for (uint i = 0; i < N; i++)
    {
        h_InputKey[i] = rand() % numValues;
        h_InputVal[i] = i;
    }

    printArray(h_InputKey, N);
    printf("Allocating and initializing CUDA arrays...\n\n");
    error = cudaMalloc((void **)&d_InputKey,  Nmax * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_InputVal,  Nmax * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_OutputKey, Nmax * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_OutputVal, Nmax * sizeof(uint));
    checkCudaErrors(error);
    
    int flag = 1;
    //printf("Running GPU bitonic sort (%u identical iterations)...\n\n", numIterations);

    
    error = cudaDeviceSynchronize();
    checkCudaErrors(error);

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
            
    uint arrayLength = Nmax;
    uint dir = DIR;
    uint threadCount;
    resetPointers();

    if(N < Nmax) {
	
        error = cudaMemcpy(d_InputKey, h1_InputKey, N * sizeof(uint), cudaMemcpyHostToDevice);
        checkCudaErrors(error);
        error = cudaMemcpy(d_InputVal, h1_InputVal, N * sizeof(uint), cudaMemcpyHostToDevice);
        checkCudaErrors(error);
        
        threadCount = bitonicSort(
            d_OutputKey,
            d_OutputVal,
            d_InputKey,
            d_InputVal,
            1,
            N,
            dir
        );

        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

        error = cudaMemcpy(h1_OutputKeyGPU, d_OutputKey, N * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);
        error = cudaMemcpy(h1_OutputValGPU, d_OutputVal, N * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);
    }
    else{
    for(int i=0; i < N/Nmax; i++){      
        printf("Testing array length %u (%u arrays per batch)...\n", arrayLength, N/arrayLength);
        
        h1_InputKey = h_InputKey + Nmax*i;
        h1_InputVal = h_InputVal + Nmax*i;
        
        error = cudaMemcpy(d_InputKey, h1_InputKey, Nmax * sizeof(uint), cudaMemcpyHostToDevice);
        checkCudaErrors(error);
        error = cudaMemcpy(d_InputVal, h1_InputVal, Nmax * sizeof(uint), cudaMemcpyHostToDevice);
        checkCudaErrors(error);
        
        threadCount = bitonicSort(
            d_OutputKey,
            d_OutputVal,
            d_InputKey,
            d_InputVal,
            1,
            Nmax,
            dir
        );

        error = cudaDeviceSynchronize();
        checkCudaErrors(error);
        h1_OutputKeyGPU = h_OutputKeyGPU + Nmax*i;
        h1_OutputValGPU = h_OutputValGPU + Nmax*i;

        error = cudaMemcpy(h1_OutputKeyGPU, d_OutputKey, Nmax * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);
        error = cudaMemcpy(h1_OutputValGPU, d_OutputVal, Nmax * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);
        
	printArray(h_OutputKeyGPU, N);
        dir = 1 - dir;
    }

    resetPointers();
    printf("Nmax done \n");
    printArray(h_OutputKeyGPU, N);
    for(arrayLength = 2*Nmax; arrayLength <= N; arrayLength*=2){
        
        for(uint size = arrayLength, stride = arrayLength/2; size > Nmax; size >>= 1, stride >>= 1){
	    printf("Performing Bitonic Merge using Nmax at a time, size: %u, stride: %u \n", size, stride);
            resetPointers();
	    dir = DIR;
            for(uint i=0; i < N/size; i++){
                
                for(uint j=0; j < stride/(Nmax/2); j++){
		    resetPointers();
                    h1_OutputKeyGPU = h_OutputKeyGPU + i*size + j*(Nmax/2);
                    h1_OutputValGPU = h_OutputValGPU + i*size + j*(Nmax/2);

                    error = cudaMemcpy(d_InputKey, h1_OutputKeyGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice);
                    checkCudaErrors(error);
                    error = cudaMemcpy(d_InputVal, h1_OutputValGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice);
                    checkCudaErrors(error);

                    error = cudaMemcpy(temp, d_InputKey, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                    checkCudaErrors(error);
                    
            printArray(temp, Nmax/2);
		    printArray(h1_OutputKeyGPU, Nmax/2); 
                   
		    h1_OutputKeyGPU += stride;
                    h1_OutputValGPU += stride;
                    d_InputKey += Nmax/2;
                    d_InputVal += Nmax/2;
                    
                    error = cudaMemcpy(d_InputKey, h1_OutputKeyGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice);
                    checkCudaErrors(error);
                    error = cudaMemcpy(d_InputVal, h1_OutputValGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice);
                    checkCudaErrors(error);
                   
                   error = cudaMemcpy(temp, d_InputKey, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                    checkCudaErrors(error);
                    
                printArray(temp, Nmax/2); 
                
                    d_InputKey -= Nmax/2;
                    d_InputVal -= Nmax/2;
                    
		    printArray(h1_OutputKeyGPU, Nmax/2); 
                    threadCount = bitonicSort1(
                        d_OutputKey,
                        d_OutputVal,
                        d_InputKey,
                        d_InputVal,
                        1,
                        Nmax,
			Nmax,
			Nmax/2,
                        dir
                    );
                    
		    h1_OutputKeyGPU = h_OutputKeyGPU + i*size + j*(Nmax/2);
                    h1_OutputValGPU = h_OutputValGPU + i*size + j*(Nmax/2);

                    error = cudaDeviceSynchronize();
                    checkCudaErrors(error);

                    error = cudaMemcpy(h1_OutputKeyGPU, d_OutputKey, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                    checkCudaErrors(error);
                    error = cudaMemcpy(h1_OutputValGPU, d_OutputVal, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                    checkCudaErrors(error);

		    printf("Half array, i: %u, j: %u, diff: %d, dir: %u \n", i, j, h1_OutputKeyGPU-h_OutputKeyGPU, dir);
                    
		    h1_OutputKeyGPU = h_OutputKeyGPU + i*size + j*(Nmax/2) + stride;
                    h1_OutputValGPU = h_OutputValGPU + i*size + j*(Nmax/2) + stride;
                    d_OutputKey += Nmax/2;
                    d_OutputVal += Nmax/2;

                    error = cudaMemcpy(h1_OutputKeyGPU, d_OutputKey, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                    checkCudaErrors(error);
                    error = cudaMemcpy(h1_OutputValGPU, d_OutputVal, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                    checkCudaErrors(error);
 
		    printArray(h_OutputKeyGPU, N); 

                    d_OutputKey -= Nmax/2;
                    d_OutputVal -= Nmax/2;
                }
                
                if(arrayLength != N) dir = 1 - dir;
            }
        }
    }
    }
    uint numIterations = 1;
    sdkStopTimer(&hTimer);
    printf("Average time: %f ms\n\n", sdkGetTimerValue(&hTimer) / numIterations);

    arrayLength = N;
   // if (arrayLength == N)
    {
        double dTimeSecs = 1.0e-3 * sdkGetTimerValue(&hTimer) / numIterations;
        printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
               (1.0e-6 * (double)arrayLength/dTimeSecs), dTimeSecs, N, 1, threadCount);
    }

    printf("\nValidating the results...\n");
    printf("...reading back GPU results\n");


    int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, 1, arrayLength, numValues, DIR);
    int valuesFlag = validateValues(h_OutputKeyGPU, h_OutputValGPU, h_InputKey, 1, arrayLength);
    flag = flag && keysFlag && valuesFlag;

    printf("\n");

    printf("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    cudaFree(d_OutputVal);
    cudaFree(d_OutputKey);
    cudaFree(d_InputVal);
    cudaFree(d_InputKey);
    free(h_OutputValGPU);
    free(h_OutputKeyGPU);
    free(h_InputVal);
    free(h_InputKey);

    exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
}
