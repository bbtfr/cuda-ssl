NVCC      := /usr/local/cuda/bin/nvcc
INCLUDES  := -Iinclude
LIBS      := 
CFLAGS    := $(INCLUDES) $(LIBS) -arch=compute_35 -code=sm_35

LIBOBJ    := aes.o des.o sha1.o md5.o md4.o test.o



# NOCUDA    := TRUE

ifdef NOCUDA
FILES     := library-cpu/%.cu
CFLAGS    += -DNOCUDA
else
FILES     := library/%.cu
endif



all: test

test: $(LIBOBJ)
	$(NVCC) $(CFLAGS) -o $@ $(LIBOBJ)

$(LIBOBJ): %.o: $(FILES)
	$(NVCC) $(CFLAGS) -o $@ -c $<

clean:
	@rm -f *.o
	@rm -f test
