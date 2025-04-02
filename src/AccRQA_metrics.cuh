#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef RQA_METRICS
#define RQA_METRICS


class RQA_ConstParams {
public:
	static const int warp = 32;
	static const int shared_memory_size = 512;
	static const int width = 32;
};

class RQA_max : public RQA_ConstParams {
	public:
	static const int dst_type = DST_MAXIMAL;
};

class RQA_euc : public RQA_ConstParams {
	public:
	static const int dst_type = DST_EUCLIDEAN;
};

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
	unsigned long long int emb
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
	unsigned long long int emb
){
	IOtype sum = 0;
	for(int m = 0; m < emb; m++){
		IOtype A = input[i + m*tau];
		IOtype B = input[j + m*tau];
		sum += (A - B)*(A - B);
	}
	sum = sqrt(sum);
	return ( ( (threshold-sum)>=0 ? 1 : 0 ) );
}

template<typename IOtype>
__inline__ __device__ int R_element_euc_cache(
	IOtype *s_input, 
	int i, 
	int j, 
	IOtype threshold, 
	int tau, 
	int emb, 
	int cache_size
){
	IOtype sum = 0;
	for(int m = 0; m < emb; m++){
		IOtype A = s_input[i + m*tau];
		IOtype B = s_input[cache_size + j + m*tau];
		sum += (A - B)*(A - B);
	}
	sum = sqrt(sum);
	return ( ( (threshold-sum)>=0 ? 1 : 0 ) );
}


template<class const_params, typename IOtype>
__inline__ __device__ int get_RP_element(
	IOtype const* __restrict__  d_input, 
	unsigned long long int row, unsigned long long int column, 
	IOtype threshold, 
	unsigned long long int tau, unsigned long long int emb
){
	int R_value = 0;
	if(const_params::dst_type==DST_EUCLIDEAN){
		R_value= R_element_euc(
			d_input,
			row, column,
			threshold, tau, emb
		);
	}
	else if(const_params::dst_type==DST_MAXIMAL){
		R_value= R_element_max(
			d_input,
			row, column,
			threshold, tau, emb
		);
	}
	return(R_value);
}

template<class const_params, typename IOtype>
__inline__ __device__ int get_RP_element_cache(
	IOtype *s_input, 
	int row, int column, 
	IOtype threshold, 
	int tau, int emb, int cache_size
){
	int R_value = 0;
	if(const_params::dst_type==DST_EUCLIDEAN){
		R_value = R_element_euc_cache(
			s_input,
			row, // row = th_y
			column, // column = th_x
			threshold, tau, emb, cache_size
		);
	}
	else if(const_params::dst_type==DST_MAXIMAL){
		R_value = R_element_max_cache(
			s_input,
			row, 
			column, 
			threshold, 
			tau, 
			emb, 
			cache_size
		);
	}
	return(R_value);
}

#endif
