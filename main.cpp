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

void printArray(uint *a, int size){
    for(int i=0; i < size; i++) printf("%u ", a[i]);
    	printf("\n"); 

}
int v1(uint* out, uint size, uint tot_size){
    uint dir = 0;
    int flag=0;
    for(uint i=0;i<tot_size/size;i++){
        for(int j=0;j<size;j++){
		if(j!=0 and dir==0 and (out[i*size+j]>out[i*size+j-1])) {
			printf("nahi chala3 %u %u %u, %u \n", i, j, out[i*size+j], out[i*size+j-1]);
			flag=i+1; 
}
		if(j!=0 and dir==1 and (out[i*size+j]<out[i*size+j-1])) flag=i+1; 
        }
        dir=1-dir;
    }
    if(flag!=0) {
    	for(uint i=0;i<tot_size;i++) printf("%u ", out[i]);
		printf("-------------------- \n");
	printf("nahi chala 1, %u %u \n", size, flag);
   }
   flag=0;
     for(uint i=0;i<tot_size/(size);i++){
        for(int j=size/2;j<size;j++){
             if(out[j]>out[i*size+size/2-1]) flag=i+1;
        }
    }
/*    if(flag!=0) {
    	for(uint i=(flag-1)*size;i<flag*size;i++) printf("%u ", out[i]);
		printf("-------------------- \n");
	printf("nahi chala 1, %u %u \n", size, flag);
   }*/
    return 0;
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
    //const uint          Nmax = 8;
    const uint           DIR = 0;
    const uint     numValues = 65536;
    uint verbose = 0;
    if(argc>=3) atoi(argv[2]);

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
    h_InputKey     = (uint *)malloc(N * sizeof(uint));
    h_InputVal     = (uint *)malloc(N * sizeof(uint));
    h_OutputKeyGPU = (uint *)malloc(N * sizeof(uint));
    h_OutputValGPU = (uint *)malloc(N * sizeof(uint));

    srand(2001);

    for (uint i = 0; i < N; i++)
    {
        h_InputKey[i] = rand()%numValues;
        h_InputVal[i] = i;
    }
    if(verbose)
    printArray(h_InputKey, N);
    printf("Allocating and initializing CUDA arrays...\n\n");
    error = cudaMalloc((void **)&d_InputKey,  N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_InputVal,  N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_OutputKey, N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_OutputVal, N * sizeof(uint));
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
	
        error = cudaMemcpy(d_InputKey, h_InputKey, N * sizeof(uint), cudaMemcpyHostToDevice);
        checkCudaErrors(error);
        error = cudaMemcpy(d_InputVal, h_InputVal, N * sizeof(uint), cudaMemcpyHostToDevice);
        checkCudaErrors(error);
        
        threadCount = bitonicSort(
            d_OutputKey,
            d_OutputVal,
            d_InputKey,
            d_InputVal,
            N,
            0,
            dir
        );

        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

        error = cudaMemcpy(h_OutputKeyGPU, d_OutputKey, N * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);
        error = cudaMemcpy(h_OutputValGPU, d_OutputVal, N * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);
    }
    else{
        memcpy(h_OutputKeyGPU, h_InputKey, N * sizeof(uint));
        memcpy(h_OutputValGPU, h_InputVal, N * sizeof(uint));
        
	if(verbose)
	printArray(h_OutputKeyGPU, N);
        
	for(arrayLength = Nmax; arrayLength <= N; arrayLength*=2){
            for(uint size = arrayLength, stride = arrayLength/2; size >= Nmax; size >>= 1, stride >>= 1){
        	    if(verbose){
                    printf("Performing Bitonic Merge using Nmax at a time, size: %u, stride: %u \n", size, stride);
		 }
              
		    //printArray(h_OutputKeyGPU, N);
 
                
                for(uint i=0; i < N/size; i++){
                    for(uint j=0; j < stride/(Nmax/2); j++){
			dir = ((i*size + j*(Nmax/2))/arrayLength)%2;
        		if(verbose)
			printf("%u %u %u, %u %u \n", i, j, size, stride, dir);	    
                        h1_OutputKeyGPU = h_OutputKeyGPU + i*size + j*(Nmax/2);
                        h1_OutputValGPU = h_OutputValGPU + i*size + j*(Nmax/2);

                        error = cudaMemcpy(d_InputKey, h1_OutputKeyGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice);
                        checkCudaErrors(error);
                        error = cudaMemcpy(d_InputVal, h1_OutputValGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice);
                        checkCudaErrors(error);
     			
			if(verbose){
               	    printf("-------------------------------------\n");
                	    printArray(h1_OutputKeyGPU, Nmax/2); 
                            printf("Half array, i: %u, j: %u, diff: %d, dir:%u, size:%u, stride %u \n", i, j, h1_OutputKeyGPU-h_OutputKeyGPU, dir, size, stride);    
                        }
        	            h1_OutputKeyGPU += stride;
                        h1_OutputValGPU += stride;
                        d_InputKey += Nmax/2;
                        d_InputVal += Nmax/2;
                        
                        error = cudaMemcpy(d_InputKey, h1_OutputKeyGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice);
                        checkCudaErrors(error);
                        error = cudaMemcpy(d_InputVal, h1_OutputValGPU, Nmax/2 * sizeof(uint), cudaMemcpyHostToDevice);
                        checkCudaErrors(error);
                       
                        d_InputKey -= Nmax/2;
                        d_InputVal -= Nmax/2;
                        
			if(verbose) 
                        printArray(h1_OutputKeyGPU, Nmax/2); 
                                        
        	    	    uint onlyMerge = 1;
		                  
                        if(size == Nmax) onlyMerge = 0;
	                    threadCount = bitonicSort(
                            d_OutputKey,
                            d_OutputVal,
                            d_InputKey,
                            d_InputVal,
                            Nmax,	
			    onlyMerge,
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

			if(verbose)
        	            printArray(h1_OutputKeyGPU, Nmax/2); 
                        
        	            h1_OutputKeyGPU += stride;
                        h1_OutputValGPU += stride;
                        d_OutputKey += Nmax/2;
                        d_OutputVal += Nmax/2;

                        error = cudaMemcpy(h1_OutputKeyGPU, d_OutputKey, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                        checkCudaErrors(error);
                        error = cudaMemcpy(h1_OutputValGPU, d_OutputVal, Nmax/2 * sizeof(uint), cudaMemcpyDeviceToHost);
                        checkCudaErrors(error);
			if(verbose)
        	            printArray(h1_OutputKeyGPU, Nmax/2); 

                        d_OutputKey -= Nmax/2;
                        d_OutputVal -= Nmax/2;
                    }
                    if(arrayLength != N) dir = 1 - dir;
                }
    		//if(arrayLength == 16) printArray(h_OutputKeyGPU, N);
            }
		int keysFlag = v1(h_OutputKeyGPU, arrayLength, N);
        }
    }
    arrayLength = N;
    uint numIterations = 1;
    sdkStopTimer(&hTimer);
    printf("Average time: %f ms\n\n", sdkGetTimerValue(&hTimer) / numIterations);

    
    double dTimeSecs = 1.0e-3 * sdkGetTimerValue(&hTimer) / numIterations;
    printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * (double)arrayLength/dTimeSecs), dTimeSecs, N, 1, threadCount);

    printf("\nValidating the results...\n");
    printf("...reading back GPU results\n");

    int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, 1, arrayLength, numValues, DIR, verbose);
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
