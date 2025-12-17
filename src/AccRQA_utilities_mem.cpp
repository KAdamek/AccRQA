/* AccRQA_utilities_error.h and AccRQA_utilities_mem.h are based on Error and Mem data structures that were created by Fred Dulwich and are used with his permission */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/AccRQA_utilities_error.hpp"
#include "../include/AccRQA_utilities_mem.hpp"

#ifdef CUDA_FOUND
#include <cuda_runtime_api.h>
#endif

#ifdef ACCRQA_R_FOUND
	#include <Rmath.h>
	#define ACCRQA_RAND() unif_rand()
#else
	#define ACCRQA_RAND() rand()
#endif

typedef struct Mem Mem;
typedef enum MemType MemType;
typedef enum MemLocation MemLocation;

struct Mem {
	MemType type;             // Enumerated memory element type.
	MemLocation location;     // Enumerated memory address space.
	int32_t is_c_contiguous;  // True if strides are C contiguous.
	int32_t is_owner;         // True if memory is owned, false if aliased.
	int32_t is_read_only;     // True if data should be considered read only.
	int32_t num_dims;         // Number of dimensions.
	int64_t num_elements;     // Total number of elements allocated.
	int32_t ref_count;        // Reference counter.
	int64_t *shape;           // Size of each dimension, in number of elements.
	int64_t *stride;          // Stride in each dimension, in bytes (Python compatible).
	void *data;               // Data pointer.
};

// Private function.
static void mem_alloc(Mem *mem, Accrqa_Error *status) {
	mem->is_owner = 1; // Set flag to indicate ownership, as we're allocating.
	const size_t bytes = mem->num_elements*mem_type_size(mem->type);
	if (*status || bytes == 0) return;
	if (mem->location == MEM_CPU) {
		mem->data = calloc(bytes, 1);
		if (!mem->data) {
			*status = ERR_MEM_ALLOC_FAILURE;
			printf("Host memory allocation failure "
				   "(requested %zu bytes)", bytes);
			return;
		}
	} else if (mem->location == MEM_GPU) {
		#ifdef CUDA_FOUND
		const cudaError_t cuda_error = cudaMalloc(&mem->data, bytes);
		if (!mem->data || cuda_error) {
			*status = ERR_MEM_ALLOC_FAILURE;
			printf("GPU memory allocation failure: %s "
				   "(requested %zu bytes)",
				   cudaGetErrorString(cuda_error), bytes);
			if (mem->data) {
				cudaFree(mem->data);
				mem->data = 0;
			}
			return;
		}
		#else
		*status = ERR_MEM_LOCATION;
		printf("Cannot allocate GPU memory: "
			   "The processing function library was compiled without "
			   "CUDA support");
		#endif
	} else {
		*status = ERR_MEM_LOCATION;
		printf("Unsupported memory location");
	}
}

Mem *mem_create(
	MemType type,
	MemLocation location,
	int32_t num_dims,
	const int64_t *shape,
	const int64_t *stride,
	Accrqa_Error *status
) {
	Mem *mem = mem_create_wrapper(0, type, location, num_dims, shape, stride, status);
	mem_alloc(mem, status);
	return mem;
}

Mem *mem_create_wrapper(
	void *data,
	MemType type,
	MemLocation location,
	int32_t num_dims,
	const int64_t *shape,
	const int64_t *stride,
	Accrqa_Error *status
) {
	Mem *mem = (Mem *)calloc(1, sizeof(Mem));
	if (mem==NULL) {
		*status = ERR_MEM_ALLOC_FAILURE;
		printf("Failed to allocate mem.");
		return mem;
	}

	mem->data = data;
	mem->ref_count = 1;
	mem->type = type;
	mem->location = location;
	mem->num_dims = num_dims;
	const int64_t element_size = mem_type_size(type);
	if (element_size <= 0) {
		*status = ERR_DATA_TYPE;
		printf("Unsupported data type");
		return mem;
	}

	// For Python compatibility, a zero-dimensional tensor is a scalar.
	mem->num_elements = 1;
	if (num_dims == 0) return mem;

	// Store shape (and strides, if given; otherwise compute them).
	mem->shape = (int64_t *)calloc(mem->num_dims, sizeof(int64_t));
	mem->stride = (int64_t *)calloc(mem->num_dims, sizeof(int64_t));
	if (mem->shape == NULL) {
		*status = ERR_MEM_ALLOC_FAILURE;
		printf("Failed to allocate mem->shape.");
		return mem;
	}
	if (mem->stride == NULL) {
		*status = ERR_MEM_ALLOC_FAILURE;
		printf("Failed to allocate mem->stride.");
		return mem;
	}
	for (int32_t i = num_dims - 1; i >= 0; --i) {
		mem->shape[i] = shape[i];
		mem->stride[i] = stride ? stride[i] : mem->num_elements*element_size;
		mem->num_elements *= shape[i];
	}

	// Check if strides are as expected for a standard contiguous C array.
	mem->is_c_contiguous = 1;
	int64_t num_elements = 1;
	for (int32_t i = num_dims - 1; i >= 0; --i) {
		if (mem_stride_dim(mem, i) != num_elements*element_size) {
			mem->is_c_contiguous = 0;
		}
		num_elements *= shape[i];
	}
	return mem;
}

Mem *mem_create_copy(
	const Mem *src,
	MemLocation location,
	Accrqa_Error *status) {
	Mem *mem = mem_create_wrapper(0, src->type, location,
								  src->num_dims, src->shape, src->stride, status);
	mem_alloc(mem, status);
	mem_copy_contents(mem, src, 0, 0, src->num_elements, status);
	return mem;
}

void mem_clear_contents(Mem *mem, Accrqa_Error *status) {
	if (*status || !mem || mem->num_elements == 0) return;
	const size_t size = mem->num_elements*mem_type_size(mem->type);
	if (mem->location == MEM_CPU) {
		memset(mem->data, 0, size);
	} else if (mem->location == MEM_GPU) {
		#ifdef CUDA_FOUND
		cudaMemset(mem->data, 0, size);
		#else
		*status = ERR_MEM_LOCATION;
		printf("The processing function library was compiled "
			   "without CUDA support");
		#endif
	} else {
		*status = ERR_MEM_LOCATION;
		printf("Unsupported memory location");
	}
}

void mem_copy_contents(
	Mem *dst,
	const Mem *src,
	int64_t offset_dst,
	int64_t offset_src,
	int64_t num_elements,
	Accrqa_Error *status) {
	#ifdef CUDA_FOUND
	cudaError_t cuda_error = cudaSuccess;
	#endif
	if (*status || !dst || !src || !dst->data || !src->data) return;
	if (src->num_elements == 0 || num_elements == 0) return;
	const int64_t element_size = mem_type_size(src->type);
	const int64_t start_dst = element_size*offset_dst;
	const int64_t start_src = element_size*offset_src;
	const size_t bytes = ((size_t)element_size)*((size_t)num_elements);
	const int location_src = src->location;
	const int location_dst = dst->location;
	const void *p_src = (const void *)((const char *)(src->data) + start_src);
	void *p_dst = (void *)((char *)(dst->data) + start_dst);

	if (location_src == MEM_CPU && location_dst == MEM_CPU) {
		memcpy(p_dst, p_src, bytes);
	}
	#ifdef CUDA_FOUND
	else if (location_src == MEM_CPU && location_dst == MEM_GPU) {
		cuda_error = cudaMemcpy(p_dst, p_src, bytes, cudaMemcpyHostToDevice);
	} else if (location_src == MEM_GPU && location_dst == MEM_CPU) {
		cuda_error = cudaMemcpy(p_dst, p_src, bytes, cudaMemcpyDeviceToHost);
	} else if (location_src == MEM_GPU && location_dst == MEM_GPU) {
		cuda_error = cudaMemcpy(p_dst, p_src, bytes, cudaMemcpyDeviceToDevice);
	}
	#endif
	else {
		*status = ERR_MEM_LOCATION;
		printf("Unsupported memory location");
	}
	#ifdef CUDA_FOUND
	if (cuda_error != cudaSuccess) {
		*status = ERR_MEM_COPY_FAILURE;
		printf("cudaMemcpy error: %s",
			   cudaGetErrorString(cuda_error));
	}
	#endif
}

void *mem_data(Mem *mem) {
	return (!mem) ? 0 : mem->data;
}

const void *mem_data_const(const Mem *mem) {
	return (!mem) ? 0 : mem->data;
}

void *mem_gpu_buffer(Mem *mem, Accrqa_Error *status) {
	if (*status || !mem) return 0;
	if (mem->location != MEM_GPU) {
		*status = ERR_MEM_LOCATION;
		printf("Requested buffer is not in GPU memory");
		return 0;
	}
	return &mem->data;
}

const void *mem_gpu_buffer_const(const Mem *mem, Accrqa_Error *status) {
	if (*status || !mem) return 0;
	if (mem->location != MEM_GPU) {
		*status = ERR_MEM_LOCATION;
		printf("Requested buffer is not in GPU memory");
		return 0;
	}
	return &mem->data;
}

void mem_free(Mem *mem) {
	if (!mem) return;
	if (--mem->ref_count > 0) return;
	if (mem->is_owner && mem->data) {
		if (mem->location == MEM_CPU) {
			free(mem->data);
		} else if (mem->location == MEM_GPU) {
			#ifdef CUDA_FOUND
			cudaFree(mem->data);
			#endif
		}
	}
	free(mem->shape);
	free(mem->stride);
	free(mem);
}

int32_t mem_is_c_contiguous(const Mem *mem) {
	return (!mem || !mem->data) ? 0 : mem->is_c_contiguous;
}

int32_t mem_is_complex(const Mem *mem) {
	return (!mem || !mem->data) ? 0 :
		(mem->type & MEM_COMPLEX) == MEM_COMPLEX;
}

int32_t mem_is_read_only(const Mem *mem) {
	return (!mem || !mem->data) ? 1 : mem->is_read_only;
}

MemLocation mem_location(const Mem *mem) {
	return (!mem) ? MEM_CPU : mem->location;
}

int32_t mem_num_dims(const Mem *mem) {
	return (!mem) ? 0 : mem->num_dims;
}

int64_t mem_num_elements(const Mem *mem) {
	return (!mem || !mem->data) ? 0 : mem->num_elements;
}

void mem_random_fill(Mem *mem, Accrqa_Error *status) {
	if (*status) return;
	if (mem->location != MEM_CPU) {
		*status = ERR_MEM_LOCATION;
		printf("Unsupported memory location");
		return;
	}
	int64_t num_elements = mem->num_elements;
	const MemType precision = mem_type_from_int(mem->type & 0x0F);
	if (mem_is_complex(mem)) num_elements *= 2;
	if (precision == MEM_FLOAT) {
		float *data = (float *)mem->data;
		for (int64_t i = 0; i < num_elements; ++i) {
			// NOLINTNEXTLINE: rand() is not a problem for our use case.
			data[i] = (float)ACCRQA_RAND() / (float)RAND_MAX;
		}
	} else if (precision == MEM_DOUBLE) {
		double *data = (double *)mem->data;
		for (int64_t i = 0; i < num_elements; ++i) {
			// NOLINTNEXTLINE: rand() is not a problem for our use case.
			data[i] = (double)ACCRQA_RAND() / (double)RAND_MAX;
		}
	}
}

void mem_ref_dec(Mem *mem) {
	mem_free(mem);
}

Mem *mem_ref_inc(Mem *mem) {
	if (!mem) return 0;
	mem->ref_count++;
	return mem;
}

void mem_set_read_only(Mem *mem, int32_t value) {
	if (!mem) return;
	mem->is_read_only = value;
}

int64_t mem_shape_dim(const Mem *mem, int32_t dim) {
	return (!mem || dim < 0 || dim >= mem->num_dims) ? 0 : mem->shape[dim];
}

int64_t mem_stride_dim(const Mem *mem, int32_t dim) {
	return (!mem || dim < 0 || dim >= mem->num_dims) ? 0 : mem->stride[dim];
}

MemType mem_type(const Mem *mem) {
	return (!mem) ? MEM_VOID : mem->type;
}

int64_t mem_type_size(MemType type) {
	switch (type) {
		case MEM_CHAR:
			return sizeof(char);
		case MEM_INT:
			return sizeof(int);
		case MEM_FLOAT:
			return sizeof(float);
		case MEM_DOUBLE:
			return sizeof(double);
		case MEM_COMPLEX_FLOAT:
			return 2*sizeof(float);
		case MEM_COMPLEX_DOUBLE:
			return 2*sizeof(double);
		default:
			return 0;
	}
}

MemType mem_type_from_int(int32_t type) {
	switch (type) {
		case(MEM_VOID): return(MEM_VOID);
		case(MEM_CHAR): return(MEM_CHAR);
		case(MEM_INT): return(MEM_INT);
		case(MEM_FLOAT): return(MEM_FLOAT);
		case(MEM_DOUBLE): return(MEM_DOUBLE);
		case(MEM_COMPLEX): return(MEM_COMPLEX);
		case(MEM_COMPLEX_FLOAT): return(MEM_COMPLEX_FLOAT);
		case(MEM_COMPLEX_DOUBLE): return(MEM_COMPLEX_DOUBLE);
		default: return(MEM_VOID);
	}
}

int32_t mem_assert(Mem *mem, MemType type, MemLocation location) {
	if (mem!=NULL) {
		if (type == mem->type && location == mem->location) return (0);
		else return (1);
	}
	return (1);
}

int32_t mem_assert_type(Mem* A, Mem* B, Accrqa_Error *status){
	if (A!=NULL && B!=NULL) {
		if (mem_type(A) != mem_type(B)) {
			*status = ERR_DATA_TYPE;
			return(1);
		}
		else return(0);
	}
	*status = ERR_MEM_ALLOC_FAILURE;
	return(1);
}

int32_t mem_assert_location(Mem* A, Mem* B, Accrqa_Error *status){
	if (A!=NULL && B!=NULL) {
		if (mem_location(A) != mem_location(B)) {
			*status = ERR_MEM_LOCATION;
			return(1);
		}
		else return(0);
	}
	*status = ERR_MEM_ALLOC_FAILURE;
	return(1);
}

int32_t mem_assert_nElements(Mem* A, Mem* B, Accrqa_Error *status){
	if (A!=NULL && B!=NULL) {
		if (mem_num_elements(A) != mem_num_elements(B)) {
			*status = ERR_INVALID_ARGUMENT;
			return(1);
		}
		else return(0);
	}
	*status = ERR_MEM_ALLOC_FAILURE;
	return(1);
}

int32_t mem_assert_null(Mem* A, Accrqa_Error *status) {
	if (A==NULL) {
		*status = ERR_MEM_ALLOC_FAILURE;
		return(1);
	}
	return(0);
}

