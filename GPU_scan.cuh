#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

class Scan_Params {
public:
	static const int warp = 32;
};

class Scan_inclusive : public Scan_Params {
public:
	static const int inclusive = 1;
};

class Scan_exclusive : public Scan_Params {
public:
	static const int inclusive = 0;
};

template<typename intype>
__inline__ __device__ intype warp_scan_inclusive(intype value, unsigned int mask){
	int local_id = threadIdx.x&(warpSize-1);
	
	#pragma unroll
	for (int i=1; i<=warpSize; i=i<<1) {
        intype temp = __shfl_up_sync(mask, value, i);
        if (local_id >= i) value = value + temp;
		__syncwarp(mask);
    }
	return value;
}

template<typename intype>
__inline__ __device__ intype warp_scan_exclusive(intype value, unsigned int mask){
	intype initial_value = value;
	value = warp_scan_inclusive(value, mask);
	return (value - initial_value);
}

template<typename intype>
__inline__ __device__ intype threadblock_scan_inclusive(intype value, intype *warp_scan_partial_sums){
	if(threadIdx.x<warpSize) warp_scan_partial_sums[threadIdx.x] = 0;
	__syncthreads();
	
	intype scan_result;
	int warp_id = threadIdx.x/warpSize;
	
	// Scan within a warp Hillis/Steel
	scan_result = warp_scan_inclusive(value, 0xffffffff);
	
	// to calculate scan for whole thread block we need to add largest values to the individual warp scans
	int local_id = threadIdx.x&(warpSize-1);
	if(local_id == (warpSize-1)) {
		warp_scan_partial_sums[warp_id] = scan_result;
	}
	
	__syncthreads();
	
	if(threadIdx.x < warpSize) {
		value = warp_scan_partial_sums[threadIdx.x];
		intype temp = warp_scan_inclusive(value, 0xffffffff);
		__syncwarp(0xffffffff);
		warp_scan_partial_sums[threadIdx.x] = temp;
	}
	
	__syncthreads();
	
	if(warp_id > 0) {
		scan_result = scan_result + warp_scan_partial_sums[warp_id-1];
	}
	
	return(scan_result);
}

template<typename intype>
__inline__ __device__  intype threadblock_scan_exclusive(intype value, intype *warp_scan_partial_sums){
	intype initial_value = value;
	value = threadblock_scan_inclusive(value, warp_scan_partial_sums);
	return (value - initial_value);
}

template<typename intype>
__inline__ __device__ intype compact(intype *s_array, intype predicate, intype value, intype *warp_scan_partial_sums, int shift){
	int address = threadblock_scan_exclusive(predicate, warp_scan_partial_sums) + shift;
	if(predicate){
		s_array[address] = value;
	}
	return(address);
}

template<class const_params, typename intype>
__global__ void GPU_scan_warp(intype *d_output, intype *d_input, unsigned int nElements, intype *d_partial_sums=NULL) {
	__shared__ intype warp_scan_partial_sums[const_params::warp];
	
	intype value, scan_result;
	
	// Loading data
	unsigned int gl_index = blockIdx.x*blockDim.x + threadIdx.x;
	if(gl_index < nElements) value = d_input[blockIdx.y*nElements + gl_index];
	else value = 0;
	
	if(const_params::inclusive) scan_result = threadblock_scan_inclusive(value, warp_scan_partial_sums);
	else                        scan_result = threadblock_scan_exclusive(value, warp_scan_partial_sums);
	
	// write-out the result
	if(gl_index < nElements) d_output[blockIdx.y*nElements + gl_index] = scan_result;
	
	// if scan is longer then one threadblock write-out final sum as well
	if (d_partial_sums != NULL && threadIdx.x == blockDim.x-1) {
        d_partial_sums[blockIdx.y*gridDim.x + blockIdx.x] = scan_result;
    }
}

template<class const_params, typename intype>
__global__ void GPU_scan_grid_followup(intype *d_output, intype *d_input, intype *d_partial_sums, unsigned int nElements){
	__shared__ intype warp_scan_partial_sums[const_params::warp];
	__shared__ intype global_scan_value;
	intype value, scan_result;
	
	// Scan of the previous values
	if(threadIdx.x < gridDim.x) {
		value = d_partial_sums[blockIdx.y*gridDim.x + threadIdx.x];
	}
	else value = 0;
	scan_result = threadblock_scan_inclusive(value, warp_scan_partial_sums);
	if( threadIdx.x == (blockIdx.x-1) ) {
		global_scan_value = scan_result;
	}
	__syncthreads();
	
	// Loading data
	int gl_index = blockIdx.x*blockDim.x + threadIdx.x;
	if(gl_index < nElements) value = d_output[blockIdx.y*nElements + gl_index];
	else value = 0;
	
	if(blockIdx.x>0){
		value = value + global_scan_value;
	}
	
	if(gl_index < nElements) {
		if(const_params::inclusive) d_output[blockIdx.y*nElements + gl_index] = value;
		else                        d_output[blockIdx.y*nElements + gl_index] = value - d_input[blockIdx.y*nElements + gl_index];
	}
}

