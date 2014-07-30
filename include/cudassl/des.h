#ifndef CUDASSL_DES_H
#define CUDASSL_DES_H

#include "common.h"

#define DES_ENCRYPT     1
#define DES_DECRYPT     0

#define CUDASSL_ERR_DES_INVALID_INPUT_LENGTH              -0x0032  /**< The data input has an invalid length. */

#define DES_BLOCK_SIZE    8

#define DES_KEY_SIZE      8
#define DES_KEY_SIZE_64   8

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          DES context structure
 */
typedef struct
{
    int mode;                   /*!<  encrypt/decrypt   */
    uint32_t sk[32];            /*!<  DES subkeys       */
}
des_context;

/**
 * \brief          Triple-DES context structure
 */
typedef struct
{
    int mode;                   /*!<  encrypt/decrypt   */
    uint32_t sk[96];            /*!<  3DES subkeys      */
}
des3_context;

/**
 * \brief          Initialize DES context
 *
 * \param ctx      DES context to be initialized
 */
void des_init(des_context *ctx);

/**
 * \brief          Clear DES context
 *
 * \param ctx      DES context to be cleared
 */
void des_free(des_context *ctx);

/**
 * \brief          Initialize Triple-DES context
 *
 * \param ctx      DES3 context to be initialized
 */
void des3_init(des3_context *ctx);

/**
 * \brief          Clear Triple-DES context
 *
 * \param ctx      DES3 context to be cleared
 */
void des3_free(des3_context *ctx);

/**
 * \brief          Set key parity on the given key to odd.
 *
 *                 DES keys are 56 bits long, but each byte is padded with
 *                 a parity bit to allow verification.
 *
 * \param key      8-byte secret key
 */
void des_key_set_parity(unsigned char key[DES_KEY_SIZE]);

/**
 * \brief          Check that key parity on the given key is odd.
 *
 *                 DES keys are 56 bits long, but each byte is padded with
 *                 a parity bit to allow verification.
 *
 * \param key      8-byte secret key
 *
 * \return         0 is parity was ok, 1 if parity was not correct.
 */
int des_key_check_key_parity(const unsigned char key[DES_KEY_SIZE]);

/**
 * \brief          Check that key is not a weak or semi-weak DES key
 *
 * \param key      8-byte secret key
 *
 * \return         0 if no weak key was found, 1 if a weak key was identified.
 */
int des_key_check_weak(const unsigned char key[DES_KEY_SIZE]);

/**
 * \brief          DES key schedule (56-bit, encryption)
 *
 * \param ctx      DES context to be initialized
 * \param key      8-byte secret key
 *
 * \return         0
 */
int des_setkey_enc(des_context *ctx, const unsigned char key[DES_KEY_SIZE]);

/**
 * \brief          DES key schedule (56-bit, decryption)
 *
 * \param ctx      DES context to be initialized
 * \param key      8-byte secret key
 *
 * \return         0
 */
int des_setkey_dec(des_context *ctx, const unsigned char key[DES_KEY_SIZE]);

/**
 * \brief          Triple-DES key schedule (112-bit, encryption)
 *
 * \param ctx      3DES context to be initialized
 * \param key      16-byte secret key
 *
 * \return         0
 */
int des3_set2key_enc(des3_context *ctx,
                      const unsigned char key[DES_KEY_SIZE * 2]);

/**
 * \brief          Triple-DES key schedule (112-bit, decryption)
 *
 * \param ctx      3DES context to be initialized
 * \param key      16-byte secret key
 *
 * \return         0
 */
int des3_set2key_dec(des3_context *ctx,
                      const unsigned char key[DES_KEY_SIZE * 2]);

/**
 * \brief          Triple-DES key schedule (168-bit, encryption)
 *
 * \param ctx      3DES context to be initialized
 * \param key      24-byte secret key
 *
 * \return         0
 */
int des3_set3key_enc(des3_context *ctx,
                      const unsigned char key[DES_KEY_SIZE * 3]);

/**
 * \brief          Triple-DES key schedule (168-bit, decryption)
 *
 * \param ctx      3DES context to be initialized
 * \param key      24-byte secret key
 *
 * \return         0
 */
int des3_set3key_dec(des3_context *ctx,
                      const unsigned char key[DES_KEY_SIZE * 3]);

#ifndef NOCUDA
int des_transfer_context(des_context *ctx);
int des3_transfer_context(des3_context *ctx);
#endif

/**
 * \brief          DES-ECB block encryption/decryption
 *
 * \param ctx      DES context
 * \param input    64-bit input block
 * \param output   64-bit output block
 *
 * \return         0 if successful
 */
#ifdef NOCUDA
int des_crypt_ecb(des_context *ctx,
                    const unsigned char input[8],
                    unsigned char output[8]);
#else
int des_crypt_ecb(const unsigned char *input,
                  size_t length,
                  unsigned char *output,
                  cuda_device *device);
#endif

#if defined(CUDASSL_CIPHER_MODE_CBC)
/**
 * \brief          DES-CBC buffer encryption/decryption
 *
 * \param ctx      DES context
 * \param mode     DES_ENCRYPT or DES_DECRYPT
 * \param length   length of the input data
 * \param iv       initialization vector (updated after use)
 * \param input    buffer holding the input data
 * \param output   buffer holding the output data
 */
int des_crypt_cbc(des_context *ctx,
                    int mode,
                    size_t length,
                    unsigned char iv[8],
                    const unsigned char *input,
                    unsigned char *output);
#endif /* CUDASSL_CIPHER_MODE_CBC */

/**
 * \brief          3DES-ECB block encryption/decryption
 *
 * \param ctx      3DES context
 * \param input    64-bit input block
 * \param output   64-bit output block
 *
 * \return         0 if successful
 */
int des3_crypt_ecb(des3_context *ctx,
                     const unsigned char input[8],
                     unsigned char output[8]);

#if defined(CUDASSL_CIPHER_MODE_CBC)
/**
 * \brief          3DES-CBC buffer encryption/decryption
 *
 * \param ctx      3DES context
 * \param mode     DES_ENCRYPT or DES_DECRYPT
 * \param length   length of the input data
 * \param iv       initialization vector (updated after use)
 * \param input    buffer holding the input data
 * \param output   buffer holding the output data
 *
 * \return         0 if successful, or CUDASSL_ERR_DES_INVALID_INPUT_LENGTH
 */
int des3_crypt_cbc(des3_context *ctx,
                     int mode,
                     size_t length,
                     unsigned char iv[8],
                     const unsigned char *input,
                     unsigned char *output);
#endif /* CUDASSL_CIPHER_MODE_CBC */

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 */
#ifdef NOCUDA
int des_self_test(int verbose);
#else
int des_self_test(int verbose, cuda_device *device);
int des_performance_test(int verbose, cuda_device *device);
int des3_performance_test(int verbose, cuda_device *device);
#endif

#ifdef __cplusplus
}
#endif

#endif /* des.h */
