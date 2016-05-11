CUDA_PATH       ?= /usr/local/cuda-7.5

# CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_FLAGS   ?= $(GENCODE_SM20)

all: cuadrado_vectores suma_vectores vectorAdd

cuadrado_vectores: cuadrado_vectores.cu
	$(CUDA_PATH)/bin/nvcc -O3 -m64 $(GENCODE_FLAGS) cuadrado_vectores.cu -o cuadrado_vectores

suma_vectores: suma_vectores.cu
	$(CUDA_PATH)/bin/nvcc -O3 -m64 $(GENCODE_FLAGS) suma_vectores.cu -o suma_vectores

vectorAdd: vectorAdd.cu
	$(CUDA_PATH)/bin/nvcc -O3 -m64 $(GENCODE_FLAGS) vectorAdd.cu -o vectorAdd

clean:
	rm -f cuadrado_vectores suma_vectores vectorAdd *.o. *.bin
