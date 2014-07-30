#ifndef CUDASSL_COMMON_H
#define CUDASSL_COMMON_H

#if !defined(CUDASSL_CONFIG_FILE)
#include "config.h"
#else
#include CUDASSL_CONFIG_FILE
#endif

#if !defined(CUDASSL_PLATFORM_FILE)
#include "platform.h"
#else
#include CUDASSL_PLATFORM_FILE
#endif

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef NOCUDA
// includes, project
#include "helpers/helper_cuda.h"
#include "helpers/helper_functions.h" // helper utility functions 

typedef struct {
  unsigned char *device_data_in;
  unsigned char *device_data_out;
  int devID;
  cudaDeviceProp prop;
} cuda_device;

#else
typedef struct {} cuda_device;
#endif

#ifndef MAX_THREAD
  #define MAX_THREAD      256
#endif
#ifndef MAX_CHUNK_SIZE
  #define MAX_CHUNK_SIZE  8*1024*1024
#endif

#define desc_data(data, len) \
  printf(#data ": "); \
  for (int i = 0; i < len; i++) \
    printf("%.2X ", ((unsigned char *)data)[i]); \
  printf("\n");


// create cuda event handles
// asynchronously issue work to the GPU (all to stream 0)
#define CUDA_START_TIME \
  cudaEvent_t start, stop; \
  checkCudaErrors(cudaEventCreate(&start)); \
  checkCudaErrors(cudaEventCreate(&stop)); \
  StopWatchInterface *timer = NULL; \
  sdkCreateTimer(&timer); \
  sdkResetTimer(&timer); \
  sdkStartTimer(&timer); \
  checkCudaErrors(cudaDeviceSynchronize()); \
  cudaEventRecord(start, 0);

#define CUDA_STOP_TIME(NAME) \
  cudaEventRecord(stop, 0); \
  sdkStopTimer(&timer); \
  checkCudaErrors(cudaThreadSynchronize()); \
  float gpu_time; \
  checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop)); \
  printf(NAME ": Time spent executing by the GPU: %.2f\n", gpu_time); \
  printf(NAME ": Time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer)); \
  checkCudaErrors(cudaEventDestroy(start)); \
  checkCudaErrors(cudaEventDestroy(stop));

#define TALK_LIKE_A_HUMAN_BEING(SIZE, PREFIX, SUFFIX) \
  i = 0; h = SIZE; \
  while(h / 1024 > 1) h /= 1024, i++; \
  printf(PREFIX "%f %cBytes" SUFFIX, h, " KMGP"[i]);

#define cuda_upload_data(input, deviceMem, size) \
  checkCudaErrors(cudaMemcpy(deviceMem, input, size, cudaMemcpyHostToDevice));

#define cuda_download_data(output, deviceMem, size) \
  checkCudaErrors(cudaMemcpy(output, deviceMem, size, cudaMemcpyDeviceToHost));

#define cuda_upload_symbol(input, symbol, size) \
  checkCudaErrors(cudaMemcpyToSymbol(symbol, input, size, 0, cudaMemcpyHostToDevice));

#ifndef TX
  #define TX (blockIdx.x * blockDim.x + threadIdx.x)
#endif // TX

#endif // CUDASSL_COMMON_H
