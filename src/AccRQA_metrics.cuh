#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef RQA_METRICS
#define RQA_METRICS

template<typename IOtype>
__inline__ __device__ int R_element_cartesian(IOtype i, IOtype j, IOtype threshold){
	return ( (fabsf(i - j)<=threshold ? 1 : 0 ) );
}

template<typename IOtype>
__inline__ __device__ int R_element_max(
		IOtype const* __restrict__ input, 
		unsigned long long int i, 
		unsigned long long int j, 
		IOtype threshold, 
		unsigned long long int tau, 
		unsigned long long int emb, 
		unsigned long long int matrix_size
	){
	IOtype max = 0;
	for(int m = 0; m < emb; m++){
		IOtype A = input[i + m*tau];
		IOtype B = input[j + m*tau];
		IOtype dist = fabsf(A - B);
		if(dist > max) max = dist;
	}
	return ( ( (threshold-max)>=0 ? 1 : 0 ) );
}

template<typename IOtype>
__inline__ __device__ int R_element_max_cache(
		IOtype *s_input, 
		int i, 
		int j, 
		IOtype threshold, 
		int tau, 
		int emb,
		int cache_size
	){
	IOtype max = 0;
	for(int m = 0; m < emb; m++){
		IOtype A = s_input[i + m*tau];
		IOtype B = s_input[cache_size + j + m*tau];
		IOtype dist = fabsf(A - B);
		if(dist > max) max = dist;
	}
	return ( ( (threshold-max)>=0 ? 1 : 0 ) );
}

template<typename IOtype>
__inline__ __device__ int R_element_euc(
		IOtype const* __restrict__ input, 
		unsigned long long int i, 
		unsigned long long int j, 
		IOtype threshold, 
		unsigned long long int tau, 
		unsigned long long int emb, 
		unsigned long long int matrix_size
	){
	IOtype sum = 0;
	//if( i > matrix_size || i < 0 || j > matrix_size || j < 0 ) return (0);
	for(int m = 0; m < emb; m++){
		IOtype A = input[i + m*tau];
		IOtype B = input[j + m*tau];
		sum += (A - B)*(A - B);
	}
	sum = sqrt(sum);
	return ( ( (threshold-max)>=0 ? 1 : 0 ) );
}

#endif
