#include <cudassl/aes.h>
#include <cudassl/des.h>
#include <cudassl/sha1.h>
#include <cudassl/md5.h>
#include <cudassl/md4.h>

int main(int argc, char const *argv[]) {

  int verbose = 1;

  aes_self_test(verbose);
  des_self_test(verbose);
  sha1_self_test(verbose);
  md5_self_test(verbose);
  md4_self_test(verbose);

  return 0;
}
