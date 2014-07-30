#include "cudassl/des.h"

#if defined(CUDASSL_DES_C)

/*
 * Expanded DES S-boxes
 */
__constant__ static const uint32_t SB1[64] = {
  0x01010400, 0x00000000, 0x00010000, 0x01010404,
  0x01010004, 0x00010404, 0x00000004, 0x00010000,
  0x00000400, 0x01010400, 0x01010404, 0x00000400,
  0x01000404, 0x01010004, 0x01000000, 0x00000004,
  0x00000404, 0x01000400, 0x01000400, 0x00010400,
  0x00010400, 0x01010000, 0x01010000, 0x01000404,
  0x00010004, 0x01000004, 0x01000004, 0x00010004,
  0x00000000, 0x00000404, 0x00010404, 0x01000000,
  0x00010000, 0x01010404, 0x00000004, 0x01010000,
  0x01010400, 0x01000000, 0x01000000, 0x00000400,
  0x01010004, 0x00010000, 0x00010400, 0x01000004,
  0x00000400, 0x00000004, 0x01000404, 0x00010404,
  0x01010404, 0x00010004, 0x01010000, 0x01000404,
  0x01000004, 0x00000404, 0x00010404, 0x01010400,
  0x00000404, 0x01000400, 0x01000400, 0x00000000,
  0x00010004, 0x00010400, 0x00000000, 0x01010004
};

__constant__ static const uint32_t SB2[64] = {
  0x80108020, 0x80008000, 0x00008000, 0x00108020,
  0x00100000, 0x00000020, 0x80100020, 0x80008020,
  0x80000020, 0x80108020, 0x80108000, 0x80000000,
  0x80008000, 0x00100000, 0x00000020, 0x80100020,
  0x00108000, 0x00100020, 0x80008020, 0x00000000,
  0x80000000, 0x00008000, 0x00108020, 0x80100000,
  0x00100020, 0x80000020, 0x00000000, 0x00108000,
  0x00008020, 0x80108000, 0x80100000, 0x00008020,
  0x00000000, 0x00108020, 0x80100020, 0x00100000,
  0x80008020, 0x80100000, 0x80108000, 0x00008000,
  0x80100000, 0x80008000, 0x00000020, 0x80108020,
  0x00108020, 0x00000020, 0x00008000, 0x80000000,
  0x00008020, 0x80108000, 0x00100000, 0x80000020,
  0x00100020, 0x80008020, 0x80000020, 0x00100020,
  0x00108000, 0x00000000, 0x80008000, 0x00008020,
  0x80000000, 0x80100020, 0x80108020, 0x00108000
};

__constant__ static const uint32_t SB3[64] = {
  0x00000208, 0x08020200, 0x00000000, 0x08020008,
  0x08000200, 0x00000000, 0x00020208, 0x08000200,
  0x00020008, 0x08000008, 0x08000008, 0x00020000,
  0x08020208, 0x00020008, 0x08020000, 0x00000208,
  0x08000000, 0x00000008, 0x08020200, 0x00000200,
  0x00020200, 0x08020000, 0x08020008, 0x00020208,
  0x08000208, 0x00020200, 0x00020000, 0x08000208,
  0x00000008, 0x08020208, 0x00000200, 0x08000000,
  0x08020200, 0x08000000, 0x00020008, 0x00000208,
  0x00020000, 0x08020200, 0x08000200, 0x00000000,
  0x00000200, 0x00020008, 0x08020208, 0x08000200,
  0x08000008, 0x00000200, 0x00000000, 0x08020008,
  0x08000208, 0x00020000, 0x08000000, 0x08020208,
  0x00000008, 0x00020208, 0x00020200, 0x08000008,
  0x08020000, 0x08000208, 0x00000208, 0x08020000,
  0x00020208, 0x00000008, 0x08020008, 0x00020200
};

__constant__ static const uint32_t SB4[64] = {
  0x00802001, 0x00002081, 0x00002081, 0x00000080,
  0x00802080, 0x00800081, 0x00800001, 0x00002001,
  0x00000000, 0x00802000, 0x00802000, 0x00802081,
  0x00000081, 0x00000000, 0x00800080, 0x00800001,
  0x00000001, 0x00002000, 0x00800000, 0x00802001,
  0x00000080, 0x00800000, 0x00002001, 0x00002080,
  0x00800081, 0x00000001, 0x00002080, 0x00800080,
  0x00002000, 0x00802080, 0x00802081, 0x00000081,
  0x00800080, 0x00800001, 0x00802000, 0x00802081,
  0x00000081, 0x00000000, 0x00000000, 0x00802000,
  0x00002080, 0x00800080, 0x00800081, 0x00000001,
  0x00802001, 0x00002081, 0x00002081, 0x00000080,
  0x00802081, 0x00000081, 0x00000001, 0x00002000,
  0x00800001, 0x00002001, 0x00802080, 0x00800081,
  0x00002001, 0x00002080, 0x00800000, 0x00802001,
  0x00000080, 0x00800000, 0x00002000, 0x00802080
};

__constant__ static const uint32_t SB5[64] = {
  0x00000100, 0x02080100, 0x02080000, 0x42000100,
  0x00080000, 0x00000100, 0x40000000, 0x02080000,
  0x40080100, 0x00080000, 0x02000100, 0x40080100,
  0x42000100, 0x42080000, 0x00080100, 0x40000000,
  0x02000000, 0x40080000, 0x40080000, 0x00000000,
  0x40000100, 0x42080100, 0x42080100, 0x02000100,
  0x42080000, 0x40000100, 0x00000000, 0x42000000,
  0x02080100, 0x02000000, 0x42000000, 0x00080100,
  0x00080000, 0x42000100, 0x00000100, 0x02000000,
  0x40000000, 0x02080000, 0x42000100, 0x40080100,
  0x02000100, 0x40000000, 0x42080000, 0x02080100,
  0x40080100, 0x00000100, 0x02000000, 0x42080000,
  0x42080100, 0x00080100, 0x42000000, 0x42080100,
  0x02080000, 0x00000000, 0x40080000, 0x42000000,
  0x00080100, 0x02000100, 0x40000100, 0x00080000,
  0x00000000, 0x40080000, 0x02080100, 0x40000100
};

__constant__ static const uint32_t SB6[64] = {
  0x20000010, 0x20400000, 0x00004000, 0x20404010,
  0x20400000, 0x00000010, 0x20404010, 0x00400000,
  0x20004000, 0x00404010, 0x00400000, 0x20000010,
  0x00400010, 0x20004000, 0x20000000, 0x00004010,
  0x00000000, 0x00400010, 0x20004010, 0x00004000,
  0x00404000, 0x20004010, 0x00000010, 0x20400010,
  0x20400010, 0x00000000, 0x00404010, 0x20404000,
  0x00004010, 0x00404000, 0x20404000, 0x20000000,
  0x20004000, 0x00000010, 0x20400010, 0x00404000,
  0x20404010, 0x00400000, 0x00004010, 0x20000010,
  0x00400000, 0x20004000, 0x20000000, 0x00004010,
  0x20000010, 0x20404010, 0x00404000, 0x20400000,
  0x00404010, 0x20404000, 0x00000000, 0x20400010,
  0x00000010, 0x00004000, 0x20400000, 0x00404010,
  0x00004000, 0x00400010, 0x20004010, 0x00000000,
  0x20404000, 0x20000000, 0x00400010, 0x20004010
};

__constant__ static const uint32_t SB7[64] = {
  0x00200000, 0x04200002, 0x04000802, 0x00000000,
  0x00000800, 0x04000802, 0x00200802, 0x04200800,
  0x04200802, 0x00200000, 0x00000000, 0x04000002,
  0x00000002, 0x04000000, 0x04200002, 0x00000802,
  0x04000800, 0x00200802, 0x00200002, 0x04000800,
  0x04000002, 0x04200000, 0x04200800, 0x00200002,
  0x04200000, 0x00000800, 0x00000802, 0x04200802,
  0x00200800, 0x00000002, 0x04000000, 0x00200800,
  0x04000000, 0x00200800, 0x00200000, 0x04000802,
  0x04000802, 0x04200002, 0x04200002, 0x00000002,
  0x00200002, 0x04000000, 0x04000800, 0x00200000,
  0x04200800, 0x00000802, 0x00200802, 0x04200800,
  0x00000802, 0x04000002, 0x04200802, 0x04200000,
  0x00200800, 0x00000000, 0x00000002, 0x04200802,
  0x00000000, 0x00200802, 0x04200000, 0x00000800,
  0x04000002, 0x04000800, 0x00000800, 0x00200002
};

__constant__ static const uint32_t SB8[64] = {
  0x10001040, 0x00001000, 0x00040000, 0x10041040,
  0x10000000, 0x10001040, 0x00000040, 0x10000000,
  0x00040040, 0x10040000, 0x10041040, 0x00041000,
  0x10041000, 0x00041040, 0x00001000, 0x00000040,
  0x10040000, 0x10000040, 0x10001000, 0x00001040,
  0x00041000, 0x00040040, 0x10040040, 0x10041000,
  0x00001040, 0x00000000, 0x00000000, 0x10040040,
  0x10000040, 0x10001000, 0x00041040, 0x00040000,
  0x00041040, 0x00040000, 0x10041000, 0x00001000,
  0x00000040, 0x10040040, 0x00001000, 0x00041040,
  0x10001000, 0x00000040, 0x10000040, 0x10040000,
  0x10040040, 0x10000000, 0x00040000, 0x10001040,
  0x00000000, 0x10041040, 0x00040040, 0x10000040,
  0x10040000, 0x10001000, 0x10001040, 0x00000000,
  0x10041040, 0x00041000, 0x00041000, 0x00001040,
  0x00001040, 0x00040040, 0x10000000, 0x10041000
};

/*
 * PC1: left and right halves bit-swap
 */
static const uint32_t LHs[16] = {
  0x00000000, 0x00000001, 0x00000100, 0x00000101,
  0x00010000, 0x00010001, 0x00010100, 0x00010101,
  0x01000000, 0x01000001, 0x01000100, 0x01000101,
  0x01010000, 0x01010001, 0x01010100, 0x01010101
};

static const uint32_t RHs[16] = {
  0x00000000, 0x01000000, 0x00010000, 0x01010000,
  0x00000100, 0x01000100, 0x00010100, 0x01010100,
  0x00000001, 0x01000001, 0x00010001, 0x01010001,
  0x00000101, 0x01000101, 0x00010101, 0x01010101,
};

/*
 * Initial Permutation macro
 */
#define DES_IP(X,Y) {                                         \
  T = ((X >>  4) ^ Y) & 0x0F0F0F0F; Y ^= T; X ^= (T <<  4);   \
  T = ((X >> 16) ^ Y) & 0x0000FFFF; Y ^= T; X ^= (T << 16);   \
  T = ((Y >>  2) ^ X) & 0x33333333; X ^= T; Y ^= (T <<  2);   \
  T = ((Y >>  8) ^ X) & 0x00FF00FF; X ^= T; Y ^= (T <<  8);   \
  Y = ((Y << 1) | (Y >> 31)) & 0xFFFFFFFF;                    \
  T = (X ^ Y) & 0xAAAAAAAA; Y ^= T; X ^= T;                   \
  X = ((X << 1) | (X >> 31)) & 0xFFFFFFFF;                    \
}

/*
 * Final Permutation macro
 */
#define DES_FP(X,Y) {                                         \
  X = ((X << 31) | (X >> 1)) & 0xFFFFFFFF;                    \
  T = (X ^ Y) & 0xAAAAAAAA; X ^= T; Y ^= T;                   \
  Y = ((Y << 31) | (Y >> 1)) & 0xFFFFFFFF;                    \
  T = ((Y >>  8) ^ X) & 0x00FF00FF; X ^= T; Y ^= (T <<  8);   \
  T = ((Y >>  2) ^ X) & 0x33333333; X ^= T; Y ^= (T <<  2);   \
  T = ((X >> 16) ^ Y) & 0x0000FFFF; Y ^= T; X ^= (T << 16);   \
  T = ((X >>  4) ^ Y) & 0x0F0F0F0F; Y ^= T; X ^= (T <<  4);   \
}

/*
 * DES round macro
 */
#define DES_ROUND(X,Y) {                      \
  T = *SK++ ^ X;                              \
  Y ^= SB8[ (T      ) & 0x3F ] ^              \
       SB6[ (T >>  8) & 0x3F ] ^              \
       SB4[ (T >> 16) & 0x3F ] ^              \
       SB2[ (T >> 24) & 0x3F ];               \
                                              \
  T = *SK++ ^ ((X << 28) | (X >> 4));         \
  Y ^= SB7[ (T      ) & 0x3F ] ^              \
       SB5[ (T >>  8) & 0x3F ] ^              \
       SB3[ (T >> 16) & 0x3F ] ^              \
       SB1[ (T >> 24) & 0x3F ];               \
}

#define SWAP(a,b) { uint32_t t = a; a = b; b = t; t = 0; }

void des_init(des_context *ctx) {
  memset(ctx, 0, sizeof(des_context));
}

void des_free(des_context *ctx) {
  if (ctx == NULL)
    return;

  zeroize(ctx, sizeof(des_context));
}

void des3_init(des3_context *ctx) {
  memset(ctx, 0, sizeof(des3_context));
}

void des3_free(des3_context *ctx) {
  if (ctx == NULL)
    return;

  zeroize(ctx, sizeof(des3_context));
}

static const unsigned char odd_parity_table[128] = { 1,  2,  4,  7,  8,
  11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38, 41, 42, 44,
  47, 49, 50, 52, 55, 56, 59, 61, 62, 64, 67, 69, 70, 73, 74, 76, 79, 81,
  82, 84, 87, 88, 91, 93, 94, 97, 98, 100, 103, 104, 107, 109, 110, 112,
  115, 117, 118, 121, 122, 124, 127, 128, 131, 133, 134, 137, 138, 140,
  143, 145, 146, 148, 151, 152, 155, 157, 158, 161, 162, 164, 167, 168,
  171, 173, 174, 176, 179, 181, 182, 185, 186, 188, 191, 193, 194, 196,
  199, 200, 203, 205, 206, 208, 211, 213, 214, 217, 218, 220, 223, 224,
  227, 229, 230, 233, 234, 236, 239, 241, 242, 244, 247, 248, 251, 253,
  254 };

void des_key_set_parity(unsigned char key[DES_KEY_SIZE]) {
  int i;

  for (i = 0; i < DES_KEY_SIZE; i++)
    key[i] = odd_parity_table[key[i] / 2];
}

/*
 * Check the given key's parity, returns 1 on failure, 0 on SUCCESS
 */
int des_key_check_key_parity(const unsigned char key[DES_KEY_SIZE]) {
  int i;

  for (i = 0; i < DES_KEY_SIZE; i++)
    if (key[i] != odd_parity_table[key[i] / 2])
      return(1);

  return(0);
}

/*
 * Table of weak and semi-weak keys
 *
 * Source: http://en.wikipedia.org/wiki/Weak_key
 *
 * Weak:
 * Alternating ones + zeros (0x0101010101010101)
 * Alternating 'F' + 'E' (0xFEFEFEFEFEFEFEFE)
 * '0xE0E0E0E0F1F1F1F1'
 * '0x1F1F1F1F0E0E0E0E'
 *
 * Semi-weak:
 * 0x011F011F010E010E and 0x1F011F010E010E01
 * 0x01E001E001F101F1 and 0xE001E001F101F101
 * 0x01FE01FE01FE01FE and 0xFE01FE01FE01FE01
 * 0x1FE01FE00EF10EF1 and 0xE01FE01FF10EF10E
 * 0x1FFE1FFE0EFE0EFE and 0xFE1FFE1FFE0EFE0E
 * 0xE0FEE0FEF1FEF1FE and 0xFEE0FEE0FEF1FEF1
 *
 */

#define WEAK_KEY_COUNT 16

static const unsigned char weak_key_table[WEAK_KEY_COUNT][DES_KEY_SIZE] = {
  { 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 },
  { 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE },
  { 0x1F, 0x1F, 0x1F, 0x1F, 0x0E, 0x0E, 0x0E, 0x0E },
  { 0xE0, 0xE0, 0xE0, 0xE0, 0xF1, 0xF1, 0xF1, 0xF1 },

  { 0x01, 0x1F, 0x01, 0x1F, 0x01, 0x0E, 0x01, 0x0E },
  { 0x1F, 0x01, 0x1F, 0x01, 0x0E, 0x01, 0x0E, 0x01 },
  { 0x01, 0xE0, 0x01, 0xE0, 0x01, 0xF1, 0x01, 0xF1 },
  { 0xE0, 0x01, 0xE0, 0x01, 0xF1, 0x01, 0xF1, 0x01 },
  { 0x01, 0xFE, 0x01, 0xFE, 0x01, 0xFE, 0x01, 0xFE },
  { 0xFE, 0x01, 0xFE, 0x01, 0xFE, 0x01, 0xFE, 0x01 },
  { 0x1F, 0xE0, 0x1F, 0xE0, 0x0E, 0xF1, 0x0E, 0xF1 },
  { 0xE0, 0x1F, 0xE0, 0x1F, 0xF1, 0x0E, 0xF1, 0x0E },
  { 0x1F, 0xFE, 0x1F, 0xFE, 0x0E, 0xFE, 0x0E, 0xFE },
  { 0xFE, 0x1F, 0xFE, 0x1F, 0xFE, 0x0E, 0xFE, 0x0E },
  { 0xE0, 0xFE, 0xE0, 0xFE, 0xF1, 0xFE, 0xF1, 0xFE },
  { 0xFE, 0xE0, 0xFE, 0xE0, 0xFE, 0xF1, 0xFE, 0xF1 }
};

int des_key_check_weak(const unsigned char key[DES_KEY_SIZE]) {
  int i;

  for (i = 0; i < WEAK_KEY_COUNT; i++)
    if (memcmp(weak_key_table[i], key, DES_KEY_SIZE) == 0)
      return(1);

  return(0);
}

static void des_setkey(uint32_t SK[32], const unsigned char key[DES_KEY_SIZE]) {
  int i;
  uint32_t X, Y, T;

  GET_UINT32_BE(X, key, 0);
  GET_UINT32_BE(Y, key, 4);

  /*
   * Permuted Choice 1
   */
  T =   ((Y >>  4) ^ X) & 0x0F0F0F0F;  X ^= T; Y ^= (T <<  4);
  T =   ((Y      ) ^ X) & 0x10101010;  X ^= T; Y ^= (T      );

  X =   (LHs[ (X      ) & 0xF] << 3) | (LHs[ (X >>  8) & 0xF ] << 2)
      | (LHs[ (X >> 16) & 0xF] << 1) | (LHs[ (X >> 24) & 0xF ]     )
      | (LHs[ (X >>  5) & 0xF] << 7) | (LHs[ (X >> 13) & 0xF ] << 6)
      | (LHs[ (X >> 21) & 0xF] << 5) | (LHs[ (X >> 29) & 0xF ] << 4);

  Y =   (RHs[ (Y >>  1) & 0xF] << 3) | (RHs[ (Y >>  9) & 0xF ] << 2)
      | (RHs[ (Y >> 17) & 0xF] << 1) | (RHs[ (Y >> 25) & 0xF ]     )
      | (RHs[ (Y >>  4) & 0xF] << 7) | (RHs[ (Y >> 12) & 0xF ] << 6)
      | (RHs[ (Y >> 20) & 0xF] << 5) | (RHs[ (Y >> 28) & 0xF ] << 4);

  X &= 0x0FFFFFFF;
  Y &= 0x0FFFFFFF;

  /*
   * calculate subkeys
   */
  for (i = 0; i < 16; i++) {
    if (i < 2 || i == 8 || i == 15) {
      X = ((X <<  1) | (X >> 27)) & 0x0FFFFFFF;
      Y = ((Y <<  1) | (Y >> 27)) & 0x0FFFFFFF;
    } else {
      X = ((X <<  2) | (X >> 26)) & 0x0FFFFFFF;
      Y = ((Y <<  2) | (Y >> 26)) & 0x0FFFFFFF;
    }

    *SK++ =   ((X <<  4) & 0x24000000) | ((X << 28) & 0x10000000)
            | ((X << 14) & 0x08000000) | ((X << 18) & 0x02080000)
            | ((X <<  6) & 0x01000000) | ((X <<  9) & 0x00200000)
            | ((X >>  1) & 0x00100000) | ((X << 10) & 0x00040000)
            | ((X <<  2) & 0x00020000) | ((X >> 10) & 0x00010000)
            | ((Y >> 13) & 0x00002000) | ((Y >>  4) & 0x00001000)
            | ((Y <<  6) & 0x00000800) | ((Y >>  1) & 0x00000400)
            | ((Y >> 14) & 0x00000200) | ((Y      ) & 0x00000100)
            | ((Y >>  5) & 0x00000020) | ((Y >> 10) & 0x00000010)
            | ((Y >>  3) & 0x00000008) | ((Y >> 18) & 0x00000004)
            | ((Y >> 26) & 0x00000002) | ((Y >> 24) & 0x00000001);

    *SK++ =   ((X << 15) & 0x20000000) | ((X << 17) & 0x10000000)
            | ((X << 10) & 0x08000000) | ((X << 22) & 0x04000000)
            | ((X >>  2) & 0x02000000) | ((X <<  1) & 0x01000000)
            | ((X << 16) & 0x00200000) | ((X << 11) & 0x00100000)
            | ((X <<  3) & 0x00080000) | ((X >>  6) & 0x00040000)
            | ((X << 15) & 0x00020000) | ((X >>  4) & 0x00010000)
            | ((Y >>  2) & 0x00002000) | ((Y <<  8) & 0x00001000)
            | ((Y >> 14) & 0x00000808) | ((Y >>  9) & 0x00000400)
            | ((Y      ) & 0x00000200) | ((Y <<  7) & 0x00000100)
            | ((Y >>  7) & 0x00000020) | ((Y >>  3) & 0x00000011)
            | ((Y <<  2) & 0x00000004) | ((Y >> 21) & 0x00000002);
  }
}

/*
 * DES key schedule (56-bit, encryption)
 */
int des_setkey_enc(des_context *ctx, const unsigned char key[DES_KEY_SIZE]) {
  des_setkey(ctx->sk, key);

  return(0);
}

/*
 * DES key schedule (56-bit, decryption)
 */
int des_setkey_dec(des_context *ctx, const unsigned char key[DES_KEY_SIZE]) {
  int i;

  des_setkey(ctx->sk, key);

  for (i = 0; i < 16; i += 2) {
    SWAP(ctx->sk[i    ], ctx->sk[30 - i]);
    SWAP(ctx->sk[i + 1], ctx->sk[31 - i]);
  }

  return(0);
}

static void des3_set2key(uint32_t esk[96],
  uint32_t dsk[96],
  const unsigned char key[DES_KEY_SIZE*2]) {

  int i;

  des_setkey(esk, key);
  des_setkey(dsk + 32, key + 8);

  for (i = 0; i < 32; i += 2)
  {
    dsk[i     ] = esk[30 - i];
    dsk[i +  1] = esk[31 - i];

    esk[i + 32] = dsk[62 - i];
    esk[i + 33] = dsk[63 - i];

    esk[i + 64] = esk[i    ];
    esk[i + 65] = esk[i + 1];

    dsk[i + 64] = dsk[i    ];
    dsk[i + 65] = dsk[i + 1];
  }
}

/*
 * Triple-DES key schedule (112-bit, encryption)
 */
int des3_set2key_enc(des3_context *ctx,
  const unsigned char key[DES_KEY_SIZE * 2]) {

  uint32_t sk[96];

  des3_set2key(ctx->sk, sk, key);
  zeroize(sk,  sizeof(sk));

  return(0);
}

/*
 * Triple-DES key schedule (112-bit, decryption)
 */
int des3_set2key_dec(des3_context *ctx,
  const unsigned char key[DES_KEY_SIZE * 2]) {

  uint32_t sk[96];

  des3_set2key(sk, ctx->sk, key);
  zeroize(sk,  sizeof(sk));

  return(0);
}

static void des3_set3key(uint32_t esk[96],
  uint32_t dsk[96],
  const unsigned char key[24]) {

  int i;

  des_setkey(esk, key);
  des_setkey(dsk + 32, key +  8);
  des_setkey(esk + 64, key + 16);

  for (i = 0; i < 32; i += 2) {
    dsk[i     ] = esk[94 - i];
    dsk[i +  1] = esk[95 - i];

    esk[i + 32] = dsk[62 - i];
    esk[i + 33] = dsk[63 - i];

    dsk[i + 64] = esk[30 - i];
    dsk[i + 65] = esk[31 - i];
  }
}

/*
 * Triple-DES key schedule (168-bit, encryption)
 */
int des3_set3key_enc(des3_context *ctx,
  const unsigned char key[DES_KEY_SIZE * 3]) {

  uint32_t sk[96];

  des3_set3key(ctx->sk, sk, key);
  zeroize(sk,  sizeof(sk));

  return(0);
}

/*
 * Triple-DES key schedule (168-bit, decryption)
 */
int des3_set3key_dec(des3_context *ctx,
  const unsigned char key[DES_KEY_SIZE * 3]) {

  uint32_t sk[96];

  des3_set3key(sk, ctx->sk, key);
  zeroize(sk,  sizeof(sk));

  return(0);
}

__constant__ des_context des_ctx;
__constant__ des3_context des3_ctx;

/*
 * DES-ECB block encryption/decryption
 */
__global__ void des_crypt_ecb_kernel(
  const unsigned char *inputs,
  unsigned char *outputs) {

  const unsigned char *input = inputs + TX * DES_BLOCK_SIZE;
  unsigned char *output = outputs + TX * DES_BLOCK_SIZE;

  int i;
  uint32_t X, Y, T, *SK;

  SK = des_ctx.sk;

  GET_UINT32_BE(X, input, 0);
  GET_UINT32_BE(Y, input, 4);

  DES_IP(X, Y);

  for (i = 0; i < 8; i++) {
    DES_ROUND(Y, X);
    DES_ROUND(X, Y);
  }

  DES_FP(Y, X);

  PUT_UINT32_BE(Y, output, 0);
  PUT_UINT32_BE(X, output, 4);
}

int des_transfer_context(des_context *ctx) {
  cuda_upload_symbol(ctx, des_ctx, sizeof(des_context));
  return 0;
}

int des_crypt_ecb(const unsigned char *input,
  size_t length,
  unsigned char *output,
  cuda_device *d) {

  cuda_upload_data(input, d->device_data_in, length);
  
  int grid_size = length / (MAX_THREAD * DES_BLOCK_SIZE);
  if (length % (MAX_THREAD * DES_BLOCK_SIZE) != 0)
    grid_size += 1;
  int thread_size = (length / DES_BLOCK_SIZE) < MAX_THREAD ? 
    length / DES_BLOCK_SIZE : MAX_THREAD;

  // printf("DES_KERNEL<<<%d,%d>>>\n", grid_size, thread_size);

  des_crypt_ecb_kernel<<<grid_size, thread_size>>>(d->device_data_in, d->device_data_out);

  cuda_download_data(output, d->device_data_out, length);

  return 0;
}

/*
 * 3DES-ECB block encryption/decryption
 */
__global__ void des3_crypt_ecb_kernel(
  const unsigned char *inputs,
  unsigned char *outputs) {

  const unsigned char *input = inputs + TX * DES_BLOCK_SIZE;
  unsigned char *output = outputs + TX * DES_BLOCK_SIZE;

  int i;
  uint32_t X, Y, T, *SK;

  SK = des3_ctx.sk;

  GET_UINT32_BE(X, input, 0);
  GET_UINT32_BE(Y, input, 4);

  DES_IP(X, Y);

  for (i = 0; i < 8; i++) {
    DES_ROUND(Y, X);
    DES_ROUND(X, Y);
  }

  for (i = 0; i < 8; i++) {
    DES_ROUND(X, Y);
    DES_ROUND(Y, X);
  }

  for (i = 0; i < 8; i++) {
    DES_ROUND(Y, X);
    DES_ROUND(X, Y);
  }

  DES_FP(Y, X);

  PUT_UINT32_BE(Y, output, 0);
  PUT_UINT32_BE(X, output, 4);
}

int des3_transfer_context(des3_context *ctx) {
  cuda_upload_symbol(ctx, des3_ctx, sizeof(des3_context));
  return 0;
}

int des3_crypt_ecb(
  const unsigned char *input,
  size_t length,
  unsigned char *output,
  cuda_device *d) {

  cuda_upload_data(input, d->device_data_in, length);
  
  int grid_size = length / (MAX_THREAD * DES_BLOCK_SIZE);
  if (length % (MAX_THREAD * DES_BLOCK_SIZE) != 0)
    grid_size += 1;
  int thread_size = (length / DES_BLOCK_SIZE) < MAX_THREAD ? 
    length / DES_BLOCK_SIZE : MAX_THREAD;

  // printf("DES_KERNEL<<<%d,%d>>>\n", grid_size, thread_size);

  des3_crypt_ecb_kernel<<<grid_size, thread_size>>>(d->device_data_in, d->device_data_out);

  cuda_download_data(output, d->device_data_out, length);

  return 0;
}
#if defined(CUDASSL_SELF_TEST)

#include <stdio.h>

/*
 * DES and 3DES test vectors from:
 *
 * http://csrc.nist.gov/groups/STM/cavp/documents/des/tripledes-vectors.zip
 */
static const unsigned char des3_test_keys[24] =
{
  0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
  0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01,
  0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23
};

static const unsigned char des3_test_buf[8] =
{
  0x4E, 0x6F, 0x77, 0x20, 0x69, 0x73, 0x20, 0x74
};

static const unsigned char des3_test_ecb_dec[3][8] =
{
  { 0xCD, 0xD6, 0x4F, 0x2F, 0x94, 0x27, 0xC1, 0x5D },
  { 0x69, 0x96, 0xC8, 0xFA, 0x47, 0xA2, 0xAB, 0xEB },
  { 0x83, 0x25, 0x39, 0x76, 0x44, 0x09, 0x1A, 0x0A }
};

static const unsigned char des3_test_ecb_enc[3][8] =
{
  { 0x6A, 0x2A, 0x19, 0xF4, 0x1E, 0xCA, 0x85, 0x4B },
  { 0x03, 0xE6, 0x9F, 0x5B, 0xFA, 0x58, 0xEB, 0x42 },
  { 0xDD, 0x17, 0xE8, 0xB8, 0xB4, 0x37, 0xD2, 0x32 }
};

/*
 * Checkup routine
 */
extern "C" int des_self_test(int verbose, cuda_device *d)
{
  int i, j, u, v, ret = 0;
  des_context ctx;
  des3_context ctx3;
  unsigned char buf[MAX_THREAD][DES_BLOCK_SIZE];

  des_init(&ctx);
  des3_init(&ctx3);

  /*
   * ECB mode
   */
  for (i = 0; i < 6; i++) {
    u = i >> 1;
    v = i  & 1;

    if (verbose != 0)
      printf("  DES%c-ECB-%3d (%s): ",
       (u == 0) ? ' ' : '3', 56 + u * 56,
       (v == DES_DECRYPT) ? "dec" : "enc");

    memcpy(buf[0], des3_test_buf, DES_BLOCK_SIZE);
    memcpy(buf[1], des3_test_buf, DES_BLOCK_SIZE);

    switch (i) {
    case 0:
      des_setkey_dec(&ctx, des3_test_keys);
      des_transfer_context(&ctx);
      break;

    case 1:
      des_setkey_enc(&ctx, des3_test_keys);
      des_transfer_context(&ctx);
      break;

    case 2:
      des3_set2key_dec(&ctx3, des3_test_keys);
      des3_transfer_context(&ctx3);
      break;

    case 3:
      des3_set2key_enc(&ctx3, des3_test_keys);
      des3_transfer_context(&ctx3);
      break;

    case 4:
      des3_set3key_dec(&ctx3, des3_test_keys);
      des3_transfer_context(&ctx3);
      break;

    case 5:
      des3_set3key_enc(&ctx3, des3_test_keys);
      des3_transfer_context(&ctx3);
      break;

    default:
      return(1);
    }

    for (j = 0; j < 10000; j++) {
      if (u == 0)
        des_crypt_ecb(*buf, DES_BLOCK_SIZE * 2, *buf, d);
      else
        des3_crypt_ecb(*buf, DES_BLOCK_SIZE * 2, *buf, d);
    }

    if (v == DES_DECRYPT) {
      if (memcmp(buf[0], des3_test_ecb_dec[u], DES_BLOCK_SIZE) != 0 && 
        memcmp(buf[1], des3_test_ecb_dec[u], DES_BLOCK_SIZE) != 0) {
        if (verbose != 0)
          printf("failed\n");

        ret = 1;
        goto exit;
      }
    } else {
      if (memcmp(buf[0], des3_test_ecb_enc[u], DES_BLOCK_SIZE) != 0 &&
        memcmp(buf[1], des3_test_ecb_enc[u], DES_BLOCK_SIZE) != 0) {
        if (verbose != 0)
          printf("failed\n");

        ret = 1;
        goto exit;
      }
    }

    if (verbose != 0)
      printf("passed\n");
  }

  if (verbose != 0)
    printf("\n");

exit:
  des_free(&ctx);
  des3_free(&ctx3);

  return(ret);
}

#define DATASIZE 1000L
#define LOOPS 1000L
extern "C" int des_performance_test(int verbose, cuda_device *d) {
  unsigned char key[DES_KEY_SIZE];
  unsigned char buf[MAX_THREAD * DATASIZE][DES_BLOCK_SIZE];
  des_context ctx;
  int i; float h;

  CUDA_START_TIME

  memset(key, 0, DES_KEY_SIZE);
  memset(buf, 0, MAX_THREAD * DES_BLOCK_SIZE * DATASIZE);
  des_init(&ctx);

  des_setkey_enc(&ctx, key);
  des_transfer_context(&ctx);

  for (int i = 0; i < LOOPS; ++i)
    des_crypt_ecb(*buf, MAX_THREAD * DES_BLOCK_SIZE * DATASIZE, *buf, d);

  CUDA_STOP_TIME("  DES -ECB- 56 (enc)")
  printf("    Block Data size: %ld\n", MAX_THREAD * DES_BLOCK_SIZE * DATASIZE);
  printf("    Block Loops: %ld\n", LOOPS);

  TALK_LIKE_A_HUMAN_BEING(MAX_THREAD * DES_BLOCK_SIZE * DATASIZE * LOOPS, "    ", " in total\n");
  TALK_LIKE_A_HUMAN_BEING(MAX_THREAD * DES_BLOCK_SIZE * DATASIZE * LOOPS / gpu_time * 1000, "    ", "/sec\n");
  
  printf("    %ld loops in total\n", LOOPS * MAX_THREAD * DATASIZE);
  printf("    %f loops/sec\n", LOOPS * MAX_THREAD * DATASIZE / gpu_time * 1000);

  if (verbose != 0)
    printf("\n");

  return 0;
}

extern "C" int des3_performance_test(int verbose, cuda_device *d) {
  unsigned char key[DES_KEY_SIZE];
  unsigned char buf[MAX_THREAD * DATASIZE][DES_BLOCK_SIZE];
  des3_context ctx;
  int i; float h;

  CUDA_START_TIME

  memset(key, 0, DES_KEY_SIZE);
  memset(buf, 0, MAX_THREAD * DES_BLOCK_SIZE * DATASIZE);
  des3_init(&ctx);

  des3_set3key_enc(&ctx, key);
  des3_transfer_context(&ctx);

  for (int i = 0; i < LOOPS; ++i)
    des3_crypt_ecb(*buf, MAX_THREAD * DES_BLOCK_SIZE * DATASIZE, *buf, d);

  CUDA_STOP_TIME("  DES3-ECB-168 (enc)")
  printf("    Block Data size: %ld\n", MAX_THREAD * DES_BLOCK_SIZE * DATASIZE);
  printf("    Block Loops: %ld\n", LOOPS);

  TALK_LIKE_A_HUMAN_BEING(MAX_THREAD * DES_BLOCK_SIZE * DATASIZE * LOOPS, "    ", " in total\n");
  TALK_LIKE_A_HUMAN_BEING(MAX_THREAD * DES_BLOCK_SIZE * DATASIZE * LOOPS / gpu_time * 1000, "    ", "/sec\n");
  
  printf("    %ld loops in total\n", LOOPS * MAX_THREAD * DATASIZE);
  printf("    %f loops/sec\n", LOOPS * MAX_THREAD * DATASIZE / gpu_time * 1000);

  if (verbose != 0)
    printf("\n");

  return 0;
}
#endif /* CUDASSL_SELF_TEST */

#endif /* CUDASSL_DES_C */
