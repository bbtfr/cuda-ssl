#include <cudassl/md4.h>

#if defined(CUDASSL_MD4_C)

__device__ void md4_init(md4_context *ctx)
{
    memset(ctx, 0, sizeof(md4_context));
}

__device__ void md4_free(md4_context *ctx)
{
    if (ctx == NULL)
        return;

    zeroize(ctx, sizeof(md4_context));
}

/*
 * MD4 context setup
 */
__device__ void md4_starts(md4_context *ctx)
{
    ctx->total[0] = 0;
    ctx->total[1] = 0;

    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE;
    ctx->state[3] = 0x10325476;
}

__device__ void md4_process(md4_context *ctx, const unsigned char data[64])
{
    uint32_t X[16], A, B, C, D;

    GET_UINT32_LE(X[ 0], data,  0);
    GET_UINT32_LE(X[ 1], data,  4);
    GET_UINT32_LE(X[ 2], data,  8);
    GET_UINT32_LE(X[ 3], data, 12);
    GET_UINT32_LE(X[ 4], data, 16);
    GET_UINT32_LE(X[ 5], data, 20);
    GET_UINT32_LE(X[ 6], data, 24);
    GET_UINT32_LE(X[ 7], data, 28);
    GET_UINT32_LE(X[ 8], data, 32);
    GET_UINT32_LE(X[ 9], data, 36);
    GET_UINT32_LE(X[10], data, 40);
    GET_UINT32_LE(X[11], data, 44);
    GET_UINT32_LE(X[12], data, 48);
    GET_UINT32_LE(X[13], data, 52);
    GET_UINT32_LE(X[14], data, 56);
    GET_UINT32_LE(X[15], data, 60);

#define S(x,n) ((x << n) | ((x & 0xFFFFFFFF) >> (32 - n)))

    A = ctx->state[0];
    B = ctx->state[1];
    C = ctx->state[2];
    D = ctx->state[3];

#define F(x, y, z) ((x & y) | ((~x) & z))
#define P(a,b,c,d,x,s) { a += F(b,c,d) + x; a = S(a,s); }

    P(A, B, C, D, X[ 0],  3);
    P(D, A, B, C, X[ 1],  7);
    P(C, D, A, B, X[ 2], 11);
    P(B, C, D, A, X[ 3], 19);
    P(A, B, C, D, X[ 4],  3);
    P(D, A, B, C, X[ 5],  7);
    P(C, D, A, B, X[ 6], 11);
    P(B, C, D, A, X[ 7], 19);
    P(A, B, C, D, X[ 8],  3);
    P(D, A, B, C, X[ 9],  7);
    P(C, D, A, B, X[10], 11);
    P(B, C, D, A, X[11], 19);
    P(A, B, C, D, X[12],  3);
    P(D, A, B, C, X[13],  7);
    P(C, D, A, B, X[14], 11);
    P(B, C, D, A, X[15], 19);

#undef P
#undef F

#define F(x,y,z) ((x & y) | (x & z) | (y & z))
#define P(a,b,c,d,x,s) { a += F(b,c,d) + x + 0x5A827999; a = S(a,s); }

    P(A, B, C, D, X[ 0],  3);
    P(D, A, B, C, X[ 4],  5);
    P(C, D, A, B, X[ 8],  9);
    P(B, C, D, A, X[12], 13);
    P(A, B, C, D, X[ 1],  3);
    P(D, A, B, C, X[ 5],  5);
    P(C, D, A, B, X[ 9],  9);
    P(B, C, D, A, X[13], 13);
    P(A, B, C, D, X[ 2],  3);
    P(D, A, B, C, X[ 6],  5);
    P(C, D, A, B, X[10],  9);
    P(B, C, D, A, X[14], 13);
    P(A, B, C, D, X[ 3],  3);
    P(D, A, B, C, X[ 7],  5);
    P(C, D, A, B, X[11],  9);
    P(B, C, D, A, X[15], 13);

#undef P
#undef F

#define F(x,y,z) (x ^ y ^ z)
#define P(a,b,c,d,x,s) { a += F(b,c,d) + x + 0x6ED9EBA1; a = S(a,s); }

    P(A, B, C, D, X[ 0],  3);
    P(D, A, B, C, X[ 8],  9);
    P(C, D, A, B, X[ 4], 11);
    P(B, C, D, A, X[12], 15);
    P(A, B, C, D, X[ 2],  3);
    P(D, A, B, C, X[10],  9);
    P(C, D, A, B, X[ 6], 11);
    P(B, C, D, A, X[14], 15);
    P(A, B, C, D, X[ 1],  3);
    P(D, A, B, C, X[ 9],  9);
    P(C, D, A, B, X[ 5], 11);
    P(B, C, D, A, X[13], 15);
    P(A, B, C, D, X[ 3],  3);
    P(D, A, B, C, X[11],  9);
    P(C, D, A, B, X[ 7], 11);
    P(B, C, D, A, X[15], 15);

#undef F
#undef P

    ctx->state[0] += A;
    ctx->state[1] += B;
    ctx->state[2] += C;
    ctx->state[3] += D;
}

/*
 * MD4 process buffer
 */
__device__ void md4_update(md4_context *ctx, const unsigned char *input, size_t ilen)
{
    size_t fill;
    uint32_t left;

    if (ilen == 0)
        return;

    left = ctx->total[0] & 0x3F;
    fill = 64 - left;

    ctx->total[0] += (uint32_t) ilen;
    ctx->total[0] &= 0xFFFFFFFF;

    if (ctx->total[0] < (uint32_t) ilen)
        ctx->total[1]++;

    if (left && ilen >= fill)
    {
        memcpy((void *) (ctx->buffer + left),
                (void *) input, fill);
        md4_process(ctx, ctx->buffer);
        input += fill;
        ilen  -= fill;
        left = 0;
    }

    while(ilen >= 64)
    {
        md4_process(ctx, input);
        input += 64;
        ilen  -= 64;
    }

    if (ilen > 0)
    {
        memcpy((void *) (ctx->buffer + left),
                (void *) input, ilen);
    }
}

__constant__ static const unsigned char md4_padding[64] =
{
 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/*
 * MD4 final digest
 */
__device__ void md4_finish(md4_context *ctx, unsigned char output[16])
{
    uint32_t last, padn;
    uint32_t high, low;
    unsigned char msglen[8];

    high = (ctx->total[0] >> 29)
         | (ctx->total[1] <<  3);
    low  = (ctx->total[0] <<  3);

    PUT_UINT32_LE(low,  msglen, 0);
    PUT_UINT32_LE(high, msglen, 4);

    last = ctx->total[0] & 0x3F;
    padn = (last < 56) ? (56 - last) : (120 - last);

    md4_update(ctx, (unsigned char *) md4_padding, padn);
    md4_update(ctx, msglen, 8);

    PUT_UINT32_LE(ctx->state[0], output,  0);
    PUT_UINT32_LE(ctx->state[1], output,  4);
    PUT_UINT32_LE(ctx->state[2], output,  8);
    PUT_UINT32_LE(ctx->state[3], output, 12);
}

/*
 * output = MD4(input buffer)
 */
__device__ void md4(const unsigned char *input, size_t ilen, unsigned char output[16])
{
    md4_context ctx;

    md4_init(&ctx);
    md4_starts(&ctx);
    md4_update(&ctx, input, ilen);
    md4_finish(&ctx, output);
    md4_free(&ctx);
}

#if defined(CUDASSL_SELF_TEST)

/*
 * RFC 1320 test vectors
 */
__constant__ static unsigned char md4_test_buf[7][81] =
{
    { "" },
    { "a" },
    { "abc" },
    { "message digest" },
    { "abcdefghijklmnopqrstuvwxyz" },
    { "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" },
    { "12345678901234567890123456789012345678901234567890123456789012" \
      "345678901234567890" }
};

__constant__ static const int md4_test_buflen[7] =
{
    0, 1, 3, 14, 26, 62, 80
};

static const unsigned char md4_test_sum[7][16] =
{
    { 0x31, 0xD6, 0xCF, 0xE0, 0xD1, 0x6A, 0xE9, 0x31,
      0xB7, 0x3C, 0x59, 0xD7, 0xE0, 0xC0, 0x89, 0xC0 },
    { 0xBD, 0xE5, 0x2C, 0xB3, 0x1D, 0xE3, 0x3E, 0x46,
      0x24, 0x5E, 0x05, 0xFB, 0xDB, 0xD6, 0xFB, 0x24 },
    { 0xA4, 0x48, 0x01, 0x7A, 0xAF, 0x21, 0xD8, 0x52,
      0x5F, 0xC1, 0x0A, 0xE8, 0x7A, 0xA6, 0x72, 0x9D },
    { 0xD9, 0x13, 0x0A, 0x81, 0x64, 0x54, 0x9F, 0xE8,
      0x18, 0x87, 0x48, 0x06, 0xE1, 0xC7, 0x01, 0x4B },
    { 0xD7, 0x9E, 0x1C, 0x30, 0x8A, 0xA5, 0xBB, 0xCD,
      0xEE, 0xA8, 0xED, 0x63, 0xDF, 0x41, 0x2D, 0xA9 },
    { 0x04, 0x3F, 0x85, 0x82, 0xF2, 0x41, 0xDB, 0x35,
      0x1C, 0xE6, 0x27, 0xE1, 0x53, 0xE7, 0xF0, 0xE4 },
    { 0xE3, 0x3B, 0x4D, 0xDC, 0x9C, 0x38, 0xF2, 0x19,
      0x9C, 0x3E, 0x7B, 0x16, 0x4F, 0xCC, 0x05, 0x36 }
};

/*
 * Checkup routine
 */
__global__ void md4_self_test_kernel(unsigned char *outputs)
{
    int i;
    unsigned char *md4sum;

    for (i = 0; i < 6; i++) {
        md4sum = outputs + (TX * 6 + i) * MD4_DIGEST_LENGTH;

        md4(md4_test_buf[i], md4_test_buflen[i], md4sum);
    }
}

int md4_self_test(int verbose, cuda_device *d) {
  int i; 
  unsigned char ret[2][3][MD4_DIGEST_LENGTH];

  md4_self_test_kernel<<<1, 2>>>(d->device_data_out);
  cuda_download_data(ret, d->device_data_out, 2 * 3 * MD4_DIGEST_LENGTH);

  for (i = 0; i < 6; i++) {
    if (verbose != 0)
      printf("  MD4 test #%d: ", i + 1);

    if (memcmp(ret[0][i], md4_test_sum[i], MD4_DIGEST_LENGTH) != 0 &&
      memcmp(ret[1][i], md4_test_sum[i], MD4_DIGEST_LENGTH) != 0) {
      if (verbose != 0)
        printf("failed\n");
    } else {
      if (verbose != 0)
        printf("passed\n");
    }
  }

  if (verbose != 0)
    printf("\n");

  return 0;
}

#define DATASIZE 1000L
#define LOOPS 100000L
__global__ void md4_performance_test_kernel() {
  int i;
  unsigned char src[8];
  unsigned char ret[MD4_DIGEST_LENGTH];
  md4_context ctx;


  md4_init(&ctx);
  md4_starts(&ctx);

  memset(src, 0, 8);

  for (i = 0; i < LOOPS; i++)
    md4_update(&ctx, src, 8);

  md4_finish(&ctx, ret);
  md4_free(&ctx);
}

extern "C" int md4_performance_test(int verbose, cuda_device *d) {
  int i; float h;

  CUDA_START_TIME

  // for (i = 0; i < LOOPS; ++i) {
  md4_performance_test_kernel<<<DATASIZE, MAX_THREAD>>>();
  // }

  CUDA_STOP_TIME("  MD4")
  printf("    Block Data size: %ld\n", MAX_THREAD * 8 * DATASIZE);
  printf("    Block Loops: %ld\n", LOOPS);

  TALK_LIKE_A_HUMAN_BEING(MAX_THREAD * 8 * DATASIZE * LOOPS, "    ", " in total\n");
  TALK_LIKE_A_HUMAN_BEING(MAX_THREAD * 8 * DATASIZE * LOOPS / gpu_time * 1000, "    ", "/sec\n");
  
  printf("    %ld loops in total\n", LOOPS * MAX_THREAD * DATASIZE);
  printf("    %f loops/sec\n", LOOPS * MAX_THREAD * DATASIZE / gpu_time * 1000);

  if (verbose != 0)
    printf("\n");

  return 0;
}

#endif /* CUDASSL_SELF_TEST */

#endif /* CUDASSL_MD4_C */
