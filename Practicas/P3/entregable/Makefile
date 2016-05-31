CUDA_PATH       ?= /usr/local/cuda-7.5

# CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_FLAGS   ?= $(GENCODE_SM20)

floyd_template: floyd_template.cu Graph.cc Graph.h
	$(CUDA_PATH)/bin/nvcc -O3 -m64 $(GENCODE_FLAGS) floyd_template.cu Graph.cc -o floyd

clean:
	rm -f floyd floyd2 floyd2d *.o. *.bin
