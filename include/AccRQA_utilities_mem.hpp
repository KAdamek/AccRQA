/* AccRQA_utilities_error.h and AccRQA_utilities_mem.h are based on Error and Mem data structures that were created by Fred Dulwich and are used with his permission */

#ifndef UTILITIES_MEM_H_
#define UTILITIES_MEM_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "AccRQA_utilities_error.hpp"

#ifdef CUDA_FOUND
#include <cuda_runtime_api.h>
#endif

struct Mem;

enum MemType {
	MEM_VOID = 0,
	MEM_CHAR = 1,
	MEM_INT = 2,
	MEM_FLOAT = 4,
	MEM_DOUBLE = 8,
	MEM_COMPLEX = 32,
	MEM_COMPLEX_FLOAT = MEM_FLOAT | MEM_COMPLEX,
	MEM_COMPLEX_DOUBLE = MEM_DOUBLE | MEM_COMPLEX
};

enum MemLocation {
	MEM_CPU = 0,
	MEM_GPU = 1
};

typedef struct Mem Mem;
typedef enum MemType MemType;
typedef enum MemLocation MemLocation;

#ifdef __cplusplus
extern "C" {
#endif

	// Allocate a multi-dimensional block of memory.
	Mem *mem_create(
		MemType type,
		MemLocation location,
		int32_t num_dims,
		const int64_t *shape, 
		const int64_t *stride,
		Accrqa_Error *status);

	// Wraps a pointer to a multi-dimensional array which is owned elsewhere.
	Mem *mem_create_wrapper(
		void *data,
		MemType type,
		MemLocation location,
		int32_t num_dims,
		const int64_t *shape,
		const int64_t *stride,
		Accrqa_Error *status);

	// Create a copy of a memory block in the specified location.
	Mem *mem_create_copy(
		const Mem *src,
		MemLocation location,
		Accrqa_Error *status);

	// Clears contents of a memory block by setting all its elements to zero
	void mem_clear_contents(Mem *mem, Accrqa_Error *status);

	// Copies memory contents from one block to another.
	void mem_copy_contents(
		Mem *dst,
		const Mem *src,
		int64_t offset_dst,
		int64_t offset_src,
		int64_t num_elements,
		Accrqa_Error *status);

	// Returns a raw pointer to the memory wrapped by the handle.
	void *mem_data(Mem *mem);

	// Returns a raw const pointer to the memory wrapped by the handle.
	const void *mem_data_const(const Mem *mem);

	// Returns a pointer to the GPU buffer, if memory is on the GPU.
	void *mem_gpu_buffer(Mem *mem, Accrqa_Error *status);

	// Returns a pointer to the GPU buffer, if memory is on the GPU.
	const void *mem_gpu_buffer_const(const Mem *mem, Accrqa_Error *status);

	// Deallocate memory or decrements the reference counter.
	void mem_free(Mem *mem);

	// Returns true if the dimension strides are C contiguous.
	int32_t mem_is_c_contiguous(const Mem *mem);

	// Returns true if data elements are of complex type.
	int32_t mem_is_complex(const Mem *mem);

	// Returns true if the read-only flag is set.
	int32_t mem_is_read_only(const Mem *mem);

	// Returns the enumerated location of the memory.
	MemLocation mem_location(const Mem *mem);

	// Returns the number of dimensions in the memory block.
	int32_t mem_num_dims(const Mem *mem);

	// Returns the total number of elements in the memory block.
	int64_t mem_num_elements(const Mem *mem);

	// Fills memory with random values between 0 and 1. Useful for testing.
	void mem_random_fill(Mem *mem, Accrqa_Error *status);

	// Decrement the reference counter.
	void mem_ref_dec(Mem *mem);

	// Increment the reference counter.
	Mem *mem_ref_inc(Mem *mem);

	// Set the flag specifying whether the memory should be read-only.
	void mem_set_read_only(Mem *mem, int32_t value);

	// Returns the number of elements in the specified dimension.
	int64_t mem_shape_dim(const Mem *mem, int32_t dim);

	// Returns the stride (in bytes) of the specified dimension.
	int64_t mem_stride_dim(const Mem *mem, int32_t dim);

	// Returns the enumerated data type of the memory.
	MemType mem_type(const Mem *mem);

	// Returns the size of one element of a data type, in bytes.
	int64_t mem_type_size(MemType type);

	// Returns mem_type based on int
	MemType mem_type_from_int(int32_t type);
	
	// Asserts multiple requirements
	int32_t mem_assert(Mem *mem, MemType type, MemLocation location);

	int32_t mem_assert_type(Mem* A, Mem* B, Accrqa_Error *status);

	int32_t mem_assert_location(Mem* A, Mem* B, Accrqa_Error *status);

	int32_t mem_assert_nElements(Mem* A, Mem* B, Accrqa_Error *status);

	int32_t mem_assert_null(Mem* A, Accrqa_Error *status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
