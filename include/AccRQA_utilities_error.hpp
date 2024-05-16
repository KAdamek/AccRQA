#ifndef _ACCRQA_UTILITIES_ERRORS_H_
#define _ACCRQA_UTILITIES_ERRORS_H_

/**
 * @enum Accrqa_Error
 *
 * @brief accrqa library error codes.
 */
enum Accrqa_Error
{
	//! No error.
	SUCCESS = 0,
	
	//! Generic runtime error.
	ERR_RUNTIME = 1,
	
	//! Invalid function argument.
	ERR_INVALID_ARGUMENT = 2,
	
	//! Unsupported data type.
	ERR_DATA_TYPE = 3,
	
	//! Memory allocation failure.
	ERR_MEM_ALLOC_FAILURE = 4,
	
	//! Memory copy failure.
	ERR_MEM_COPY_FAILURE = 5,
	
	//! Unsupported memory location.
	ERR_MEM_LOCATION = 6,
	
	//! CUDA not found
	ERR_CUDA_NOT_FOUND = 7,
	
	//! CUDA device not found
	ERR_CUDA_DEVICE_NOT_FOUND = 8,
	
	//! CUDA device does not have enough memory
	ERR_CUDA_NOT_ENOUGH_MEMORY = 9,
	
	//! CUDA error
	ERR_CUDA = 10,
	
	//! Invalid metric type
	ERR_INVALID_METRIC_TYPE = 11
};

typedef enum Accrqa_Error Accrqa_Error;

#endif
