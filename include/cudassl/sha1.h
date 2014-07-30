#ifndef CUDASSL_SHA1_H
#define CUDASSL_SHA1_H

#include "common.h"

#define SHA_DIGEST_LENGTH 20

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          SHA-1 context structure
 */
typedef struct
{
    uint32_t total[2];          /*!< number of bytes processed  */
    uint32_t state[5];          /*!< intermediate digest state  */
    unsigned char buffer[64];   /*!< data block being processed */

    unsigned char ipad[64];     /*!< HMAC: inner padding        */
    unsigned char opad[64];     /*!< HMAC: outer padding        */
}
sha1_context;

#ifdef NOCUDA
/**
 * \brief          Initialize SHA-1 context
 *
 * \param ctx      SHA-1 context to be initialized
 */
void sha1_init(sha1_context *ctx);

/**
 * \brief          Clear SHA-1 context
 *
 * \param ctx      SHA-1 context to be cleared
 */
void sha1_free(sha1_context *ctx);

/**
 * \brief          SHA-1 context setup
 *
 * \param ctx      context to be initialized
 */
void sha1_starts(sha1_context *ctx);

/**
 * \brief          SHA-1 process buffer
 *
 * \param ctx      SHA-1 context
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 */
void sha1_update(sha1_context *ctx, const unsigned char *input, size_t ilen);

/**
 * \brief          SHA-1 final digest
 *
 * \param ctx      SHA-1 context
 * \param output   SHA-1 checksum result
 */
void sha1_finish(sha1_context *ctx, unsigned char output[20]);

/* Internal use */
void sha1_process(sha1_context *ctx, const unsigned char data[64]);

/**
 * \brief          Output = SHA-1(input buffer)
 *
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 * \param output   SHA-1 checksum result
 */
void sha1(const unsigned char *input, size_t ilen, unsigned char output[20]);

#endif /* NOCUDA */

/**
 * \brief          Output = SHA-1(file contents)
 *
 * \param path     input file name
 * \param output   SHA-1 checksum result
 *
 * \return         0 if successful, or CUDASSL_ERR_SHA1_FILE_IO_ERROR
 */
int sha1_file(const char *path, unsigned char output[20]);

/**
 * \brief          SHA-1 HMAC context setup
 *
 * \param ctx      HMAC context to be initialized
 * \param key      HMAC secret key
 * \param keylen   length of the HMAC key
 */
void sha1_hmac_starts(sha1_context *ctx, const unsigned char *key,
                       size_t keylen);

/**
 * \brief          SHA-1 HMAC process buffer
 *
 * \param ctx      HMAC context
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 */
void sha1_hmac_update(sha1_context *ctx, const unsigned char *input,
                       size_t ilen);

/**
 * \brief          SHA-1 HMAC final digest
 *
 * \param ctx      HMAC context
 * \param output   SHA-1 HMAC checksum result
 */
void sha1_hmac_finish(sha1_context *ctx, unsigned char output[20]);

/**
 * \brief          SHA-1 HMAC context reset
 *
 * \param ctx      HMAC context to be reset
 */
void sha1_hmac_reset(sha1_context *ctx);

/**
 * \brief          Output = HMAC-SHA-1(hmac key, input buffer)
 *
 * \param key      HMAC secret key
 * \param keylen   length of the HMAC key
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 * \param output   HMAC-SHA-1 result
 */
void sha1_hmac(const unsigned char *key, size_t keylen,
                const unsigned char *input, size_t ilen,
                unsigned char output[20]);

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 */
#ifdef NOCUDA
int sha1_self_test(int verbose);
#else
int sha1_self_test(int verbose, cuda_device *device);
int sha1_performance_test(int verbose, cuda_device *device);
#endif

#ifdef __cplusplus
}
#endif

#endif /* sha1.h */
