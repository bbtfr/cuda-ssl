#ifndef CUDASSL_PLATFORM_H
#define CUDASSL_PLATFORM_H

#include <stdio.h>

#if defined(_MSC_VER) && !defined(EFIX64) && !defined(EFI32)
#include <basetsd.h>
typedef UINT32 uint32_t;
#else
#include <inttypes.h>
#endif

/*
 * 32-bit integer manipulation macros (big endian)
 */
#ifndef GET_UINT32_BE
#define GET_UINT32_BE(n,b,i)                            \
{                                                       \
    (n) = ((uint32_t) (b)[(i)    ] << 24)             \
        | ((uint32_t) (b)[(i) + 1] << 16)             \
        | ((uint32_t) (b)[(i) + 2] <<  8)             \
        | ((uint32_t) (b)[(i) + 3]       );            \
}
#endif

#ifndef PUT_UINT32_BE
#define PUT_UINT32_BE(n,b,i)                            \
{                                                       \
    (b)[(i)    ] = (unsigned char) ((n) >> 24);       \
    (b)[(i) + 1] = (unsigned char) ((n) >> 16);       \
    (b)[(i) + 2] = (unsigned char) ((n) >>  8);       \
    (b)[(i) + 3] = (unsigned char) ((n)       );       \
}
#endif

/*
 * 32-bit integer manipulation macros (little endian)
 */
#ifndef GET_UINT32_LE
#define GET_UINT32_LE(n,b,i) {                      \
  (n) = ((uint32_t) (b)[(i)    ]      )             \
      | ((uint32_t) (b)[(i) + 1] <<  8)             \
      | ((uint32_t) (b)[(i) + 2] << 16)             \
      | ((uint32_t) (b)[(i) + 3] << 24);            \
}
#endif

#ifndef PUT_UINT32_LE
#define PUT_UINT32_LE(n,b,i) {                      \
  (b)[(i)    ] = (unsigned char) ((n)      );       \
  (b)[(i) + 1] = (unsigned char) ((n) >>  8);       \
  (b)[(i) + 2] = (unsigned char) ((n) >> 16);       \
  (b)[(i) + 3] = (unsigned char) ((n) >> 24);       \
}
#endif

#define zeroize(v, n) memset(v, 0, n)

#endif /* platform.h */
