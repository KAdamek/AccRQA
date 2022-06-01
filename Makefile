INC := -I${CUDA_HOME}/include -I/usr/share/R/include
LIB := -L${CUDA_HOME}/lib64 -L/usr/lib/R/lib -lcudart -lcuda -lR
LIBRQA := -L. -lAccRQA

# use this compilers
# g++ just because the file write
GCC = g++
NVCC = ${CUDA_HOME}/bin/nvcc

NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v -Xcompiler -Wextra -Xcompiler -fPIC -lineinfo

GCC_OPTS =-O3 -fPIC -Wall -Wextra $(INC)

ANALYZE = RQA.exe


ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

ifdef fastmath
NVCCFLAGS += --use_fast_math
endif

all: clean sharedlibrary analyze 

sharedlibrary: AccRQA-GPU-RR.o AccRQA-GPU-RR-ER.o AccRQA-GPU-LAM.o AccRQA_CPU_function.o AccRQA_RecurrentRate.o AccRQA_LengthHistogram.o AccRQA_R_bindings.o Makefile
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) -shared -o libAccRQA.so AccRQA-GPU-RR.o AccRQA-GPU-RR-ER.o AccRQA-GPU-LAM.o AccRQA_CPU_function.o AccRQA_RecurrentRate.o AccRQA_LengthHistogram.o AccRQA_R_bindings.o

analyze: Makefile
	$(GCC) $(GCC_OPTS) -o $(ANALYZE) RQA.cpp $(LIBRQA)

AccRQA-GPU-RR.o: timer.h utils_cuda.h
	$(NVCC) -c AccRQA-GPU-RR.cu $(NVCCFLAGS)

AccRQA-GPU-RR-ER.o: timer.h utils_cuda.h
	$(NVCC) -c AccRQA-GPU-RR-ER.cu $(NVCCFLAGS)

AccRQA-GPU-LAM.o: timer.h utils_cuda.h
	$(NVCC) -c AccRQA-GPU-LAM.cu $(NVCCFLAGS)

AccRQA_CPU_function.o: AccRQA_CPU_function.cpp
	$(GCC) -c AccRQA_CPU_function.cpp $(GCC_OPTS)

AccRQA_LengthHistogram.o: AccRQA_LengthHistogram.cpp
	$(GCC) -c AccRQA_LengthHistogram.cpp $(GCC_OPTS)
	
AccRQA_RecurrentRate.o: AccRQA_RecurrentRate.cpp
	$(GCC) -c AccRQA_RecurrentRate.cpp $(GCC_OPTS)
	
AccRQA_R_bindings.o: AccRQA_R_bindings.cpp
	$(GCC) -c AccRQA_R_bindings.cpp $(GCC_OPTS) $(LIB)



#RQA_RecurrentRate.o: RQA_RecurrentRate.cpp
#	$(GCC) -c RQA_RecurrentRate.cpp $(GCC_OPTS)


clean:	
	rm -f *.o *.~ $(ANALYZE)


