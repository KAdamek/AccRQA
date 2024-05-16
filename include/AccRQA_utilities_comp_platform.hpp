#ifndef _ACCRQA_UTILITIES_COMPPLATFORM_H_
#define _ACCRQA_UTILITIES_COMPPLATFORM_H_

/**
 * @enum Accrqa_CompPlatform
 *
 * @brief Enumerator to specify processing platform to be used by accrqa.
 */
enum Accrqa_CompPlatform
{
	//! Euclidean distance.
	PLT_CPU = 1,
	
	//! Maximal distance .
	PLT_NV_GPU = 1024,
};

typedef enum Accrqa_CompPlatform Accrqa_CompPlatform;

#endif
