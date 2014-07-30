#ifndef CUDASSL_AES_H
#define CUDASSL_AES_H

#include "common.h"

#define AES_ENCRYPT     1
#define AES_DECRYPT     0

#define AES_BLOCK_SIZE    16

#define AES_KEY_SIZE      32
#define AES_KEY_SIZE_128  16
#define AES_KEY_SIZE_192  24
#define AES_KEY_SIZE_256  32

#define CUDASSL_ERR_AES_INVALID_KEY_LENGTH                -0x0020  /**< Invalid key length. */
#define CUDASSL_ERR_AES_INVALID_INPUT_LENGTH              -0x0022  /**< Invalid data input length. */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          AES context structure
 *
 * \note           rk is able to hold 32 extra bytes, which can be used
 *                 to simplify key expansion in the 256-bit case by
 *                 generating an extra round key
 */
typedef struct {
    int nr;                     /*!<  number of rounds  */
    uint32_t rk[68];            /*!<  AES round key     */
} aes_context;

/**
 * \brief          Initialize AES context
 *
 * \param ctx      AES context to be initialized
 */
void aes_init(aes_context *ctx);

/**
 * \brief          Clear AES context
 *
 * \param ctx      AES context to be cleared
 */
void aes_free(aes_context *ctx);

/**
 * \brief          AES key schedule (encryption)
 *
 * \param ctx      AES context to be initialized
 * \param key      encryption key
 * \param keysize  must be 128, 192 or 256
 *
 * \return         0 if successful, or CUDASSL_ERR_AES_INVALID_KEY_LENGTH
 */
int aes_setkey_enc(aes_context *ctx, const unsigned char *key,
                    unsigned int keysize);

/**
 * \brief          AES key schedule (decryption)
 *
 * \param ctx      AES context to be initialized
 * \param key      decryption key
 * \param keysize  must be 128, 192 or 256
 *
 * \return         0 if successful, or CUDASSL_ERR_AES_INVALID_KEY_LENGTH
 */
int aes_setkey_dec(aes_context *ctx, const unsigned char *key,
                    unsigned int keysize);

#ifndef NOCUDA
int aes_transfer_context(aes_context *ctx);
#endif

/**
 * \brief          AES-ECB block encryption/decryption
 *
 * \param ctx      AES context
 * \param mode     AES_ENCRYPT or AES_DECRYPT
 * \param input    16-byte input block
 * \param output   16-byte output block
 *
 * \return         0 if successful
 */
#ifdef NOCUDA
int aes_crypt_ecb(aes_context *ctx,
                  int mode,
                  const unsigned char input[16],
                  unsigned char output[16]);
#else
int aes_crypt_ecb(int mode,
                  const unsigned char *input,
                  size_t length,
                  unsigned char *output,
                  cuda_device *device);
#endif

#if defined(CUDASSL_CIPHER_MODE_CBC)
/**
 * \brief          AES-CBC buffer encryption/decryption
 *                 Length should be a multiple of the block
 *                 size (16 bytes)
 *
 * \param ctx      AES context
 * \param mode     AES_ENCRYPT or AES_DECRYPT
 * \param length   length of the input data
 * \param iv       initialization vector (updated after use)
 * \param input    buffer holding the input data
 * \param output   buffer holding the output data
 *
 * \return         0 if successful, or CUDASSL_ERR_AES_INVALID_INPUT_LENGTH
 */
int aes_crypt_cbc(aes_context *ctx,
                    int mode,
                    size_t length,
                    unsigned char iv[16],
                    const unsigned char *input,
                    unsigned char *output);
#endif /* CUDASSL_CIPHER_MODE_CBC */

#if defined(CUDASSL_CIPHER_MODE_CFB)
/**
 * \brief          AES-CFB128 buffer encryption/decryption.
 *
 * Note: Due to the nature of CFB you should use the same key schedule for
 * both encryption and decryption. So a context initialized with
 * aes_setkey_enc() for both AES_ENCRYPT and AES_DECRYPT.
 *
 * \param ctx      AES context
 * \param mode     AES_ENCRYPT or AES_DECRYPT
 * \param length   length of the input data
 * \param iv_off   offset in IV (updated after use)
 * \param iv       initialization vector (updated after use)
 * \param input    buffer holding the input data
 * \param output   buffer holding the output data
 *
 * \return         0 if successful
 */
int aes_crypt_cfb128(aes_context *ctx,
                       int mode,
                       size_t length,
                       size_t *iv_off,
                       unsigned char iv[16],
                       const unsigned char *input,
                       unsigned char *output);

/**
 * \brief          AES-CFB8 buffer encryption/decryption.
 *
 * Note: Due to the nature of CFB you should use the same key schedule for
 * both encryption and decryption. So a context initialized with
 * aes_setkey_enc() for both AES_ENCRYPT and AES_DECRYPT.
 *
 * \param ctx      AES context
 * \param mode     AES_ENCRYPT or AES_DECRYPT
 * \param length   length of the input data
 * \param iv       initialization vector (updated after use)
 * \param input    buffer holding the input data
 * \param output   buffer holding the output data
 *
 * \return         0 if successful
 */
int aes_crypt_cfb8(aes_context *ctx,
                    int mode,
                    size_t length,
                    unsigned char iv[16],
                    const unsigned char *input,
                    unsigned char *output);
#endif /*CUDASSL_CIPHER_MODE_CFB */

#if defined(CUDASSL_CIPHER_MODE_CTR)
/**
 * \brief               AES-CTR buffer encryption/decryption
 *
 * Warning: You have to keep the maximum use of your counter in mind!
 *
 * Note: Due to the nature of CTR you should use the same key schedule for
 * both encryption and decryption. So a context initialized with
 * aes_setkey_enc() for both AES_ENCRYPT and AES_DECRYPT.
 *
 * \param ctx           AES context
 * \param length        The length of the data
 * \param nc_off        The offset in the current stream_block (for resuming
 *                      within current cipher stream). The offset pointer to
 *                      should be 0 at the start of a stream.
 * \param nonce_counter The 128-bit nonce and counter.
 * \param stream_block  The saved stream-block for resuming. Is overwritten
 *                      by the function.
 * \param input         The input data stream
 * \param output        The output data stream
 *
 * \return         0 if successful
 */
int aes_crypt_ctr(aes_context *ctx,
                       size_t length,
                       size_t *nc_off,
                       unsigned char nonce_counter[16],
                       unsigned char stream_block[16],
                       const unsigned char *input,
                       unsigned char *output);
#endif /* CUDASSL_CIPHER_MODE_CTR */

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 */
#ifdef NOCUDA
int aes_self_test(int verbose);
#else
int aes_self_test(int verbose, cuda_device *device);
int aes_performance_test(int verbose, cuda_device *device);
#endif

#ifdef __cplusplus
}
#endif

#endif /* aes.h */
