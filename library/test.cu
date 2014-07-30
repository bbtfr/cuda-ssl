#include <cudassl/aes.h>
#include <cudassl/des.h>
#include <cudassl/sha1.h>
#include <cudassl/md5.h>
#include <cudassl/md4.h>
#include <cudassl/helpers/helper_test.h>

static __device__ unsigned char *device_data_in;
static __device__ unsigned char *device_data_out;

cuda_device device = {
  device_data_in, device_data_out
};

int main_self_test(int verbose, cuda_device *d);

int main(int argc, char const *argv[]) {

  cuda_deviceInit(MAX_CHUNK_SIZE, &device);

  int verbose = 1;

  aes_self_test(verbose, &device);
  aes_performance_test(verbose, &device);
  des_self_test(verbose, &device);
  des_performance_test(verbose, &device);
  des3_performance_test(verbose, &device);
  sha1_self_test(verbose, &device);
  sha1_performance_test(verbose, &device);
  md5_self_test(verbose, &device);
  md5_performance_test(verbose, &device);
  md4_self_test(verbose, &device);
  md4_performance_test(verbose, &device);

  cuda_deviceFinish(&device);

  return 0;
}

// MAIN SELF TEST
__global__ void main_kernel(unsigned char *data) {
  data[TX*4] = threadIdx.x;
  data[TX*4+1] = blockIdx.x;
  data[TX*4+2] = blockDim.x;
  data[TX*4+3] = TX;
}

#define BLOCKDIM 4
#define TOTAL 4*BLOCKDIM*MAX_THREAD
int main_self_test(int verbose, cuda_device *d) {
  unsigned char input[TOTAL] = { 0 };
  memset(input, 0, TOTAL * sizeof(unsigned char));
  cuda_upload_data(input, device.device_data_in, TOTAL * sizeof(unsigned char));
  main_kernel<<<BLOCKDIM, MAX_THREAD>>>(device.device_data_in);
  cuda_download_data(input, device.device_data_in, TOTAL * sizeof(unsigned char));
  printf("ThreadIdx\tBlockIdx\tBlockDim\tTX\n");
  for (int i = 0; i < TOTAL / 4; i++) {
    for (int j = 0; j < 4; ++j)
    {
      printf("%d\t\t", input[i*4+j]);
    }
    printf("\n");
  }
  return 0;
}