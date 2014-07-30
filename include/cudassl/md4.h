#ifndef CUDASSL_MD4_H
#define CUDASSL_MD4_H

#include "common.h"

#define MD4_DIGEST_LENGTH 16

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          MD4 context structure
 */
typedef struct
{
    uint32_t total[2];          /*!< number of bytes processed  */
    uint32_t state[4];          /*!< intermediate digest state  */
    unsigned char buffer[64];   /*!< data block being processed */

    unsigned char ipad[64];     /*!< HMAC: inner padding        */
    unsigned char opad[64];     /*!< HMAC: outer padding        */
}
md4_context;

#ifdef NOCUDA
/**
 * \brief          Initialize MD4 context
 *
 * \param ctx      MD4 context to be initialized
 */
void md4_init(md4_context *ctx);

/**
 * \brief          Clear MD4 context
 *
 * \param ctx      MD4 context to be cleared
 */
void md4_free(md4_context *ctx);

/**
 * \brief          MD4 context setup
 *
 * \param ctx      context to be initialized
 */
void md4_starts(md4_context *ctx);

/**
 * \brief          MD4 process buffer
 *
 * \param ctx      MD4 context
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 */
void md4_update(md4_context *ctx, const unsigned char *input, size_t ilen);

/**
 * \brief          MD4 final digest
 *
 * \param ctx      MD4 context
 * \param output   MD4 checksum result
 */
void md4_finish(md4_context *ctx, unsigned char output[16]);

/* Internal use */
void md4_process(md4_context *ctx, const unsigned char data[64]);

/**
 * \brief          Output = MD4(input buffer)
 *
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 * \param output   MD4 checksum result
 */
void md4(const unsigned char *input, size_t ilen, unsigned char output[16]);

#endif /* NOCUDA */
/**
 * \brief          Output = MD4(file contents)
 *
 * \param path     input file name
 * \param output   MD4 checksum result
 *
 * \return         0 if successful, or CUDASSL_ERR_MD4_FILE_IO_ERROR
 */
int md4_file(const char *path, unsigned char output[16]);

/**
 * \brief          MD4 HMAC context setup
 *
 * \param ctx      HMAC context to be initialized
 * \param key      HMAC secret key
 * \param keylen   length of the HMAC key
 */
void md4_hmac_starts(md4_context *ctx, const unsigned char *key,
                      size_t keylen);

/**
 * \brief          MD4 HMAC process buffer
 *
 * \param ctx      HMAC context
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 */
void md4_hmac_update(md4_context *ctx, const unsigned char *input,
                      size_t ilen);

/**
 * \brief          MD4 HMAC final digest
 *
 * \param ctx      HMAC context
 * \param output   MD4 HMAC checksum result
 */
void md4_hmac_finish(md4_context *ctx, unsigned char output[16]);

/**
 * \brief          MD4 HMAC context reset
 *
 * \param ctx      HMAC context to be reset
 */
void md4_hmac_reset(md4_context *ctx);

/**
 * \brief          Output = HMAC-MD4(hmac key, input buffer)
 *
 * \param key      HMAC secret key
 * \param keylen   length of the HMAC key
 * \param input    buffer holding the  data
 * \param ilen     length of the input data
 * \param output   HMAC-MD4 result
 */
void md4_hmac(const unsigned char *key, size_t keylen,
               const unsigned char *input, size_t ilen,
               unsigned char output[16]);

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 */
#ifdef NOCUDA
int md4_self_test(int verbose);
#else
int md4_self_test(int verbose, cuda_device *device);
int md4_performance_test(int verbose, cuda_device *device);
#endif

#ifdef __cplusplus
}
#endif

#endif /* md4.h */
