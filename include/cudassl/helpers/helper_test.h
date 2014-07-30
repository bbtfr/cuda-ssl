#define OUTPUT_QUIET      0
#define OUTPUT_NORMAL     1
#define OUTPUT_VERBOSE    2
int outputVerbosity = 2;

void checkCudaDevice(cuda_device *d) {
  int deviceCount, devID = 0;

  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  if (!deviceCount) {
    fprintf(stderr, "There is no device supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  printf("Successfully found %d CUDA devices (CUDART_VERSION %d).\n", deviceCount, CUDART_VERSION);
  
  // Otherwise pick the device with highest Gflops/s
  devID = gpuGetMaxGflopsDeviceId();
  checkCudaErrors(cudaSetDevice(devID));
  checkCudaErrors(cudaGetDeviceProperties(&d->prop, devID));
  
  if (outputVerbosity == OUTPUT_VERBOSE) {
    printf("\nDevice %d: \"%s\"\n", devID, d->prop.name);
    printf("  CUDA Compute Capability:\t\t\t%d.%d\n", d->prop.major, d->prop.minor);
    printf("  Number of multiprocessors (SM):\t\t%d\n", d->prop.multiProcessorCount);
    printf("  Integrated:\t\t\t\t\t%s\n", d->prop.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:\t%s\n", d->prop.canMapHostMemory ? "Yes" : "No");
    printf("\n");
  }
}

static float timeElapsed;
static cudaEvent_t timeStart, timeStop;

extern "C" void cuda_deviceInit(int bufferSize, cuda_device *d) {
  checkCudaDevice(d);
  
  if (!bufferSize)
    bufferSize = MAX_CHUNK_SIZE;
  
  checkCudaErrors(cudaMalloc((void**)&d->device_data_in, bufferSize));
  checkCudaErrors(cudaMalloc((void**)&d->device_data_out, bufferSize));

  if (outputVerbosity != OUTPUT_QUIET)
    printf("The current buffer size is %d.\n\n", bufferSize);

  if (outputVerbosity >= OUTPUT_NORMAL) {
    checkCudaErrors(cudaEventCreate(&timeStart));
    checkCudaErrors(cudaEventCreate(&timeStop));
    checkCudaErrors(cudaEventRecord(timeStart, 0));
  }

}

extern "C" void cuda_deviceFinish(cuda_device *d) {
  if (outputVerbosity >= OUTPUT_NORMAL) 
    printf("\nDone. Finishing up...\n");

  checkCudaErrors(cudaFree(d->device_data_in));
  checkCudaErrors(cudaFree(d->device_data_out));

  if (outputVerbosity >= OUTPUT_NORMAL) {
    checkCudaErrors(cudaEventRecord(timeStop, 0));
    checkCudaErrors(cudaEventSynchronize(timeStop));
    checkCudaErrors(cudaEventElapsedTime(&timeElapsed, timeStart, timeStop));
    printf("\nTotal time: %f milliseconds\n", timeElapsed); 
  }
}
