#ifndef _ACCRQA_UTILITIES_COMPPLATFORM_H_
#define _ACCRQA_UTILITIES_COMPPLATFORM_H_

/**
 * @enum Accrqa_CompPlatform
 *
 * @brief Enumerator to specify processing platform to be used by accrqa.
 */
enum Accrqa_CompPlatform
{
	//! No computational platform
	PLT_ERROR = 0,
	
	//! CPUs will be used
	PLT_CPU = 1,
	
	//! NVIDIA GPUs will bbe used
	PLT_NV_GPU = 1024,
};

typedef enum Accrqa_CompPlatform Accrqa_CompPlatform;

#endif
