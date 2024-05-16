#ifndef _ACCRQA_UTILITIES_DISTANCE_H_
#define _ACCRQA_UTILITIES_DISTANCE_H_

/**
 * @enum Accrqa_Distance
 *
 * @brief Enumerator to specify the formula to be used for calculation of distance to the line of identity.
 */
enum Accrqa_Distance
{
	//! Euclidean distance.
	DST_EUCLIDEAN = 1,
	
	//! Maximal distance .
	DST_MAXIMAL = 2,
};

typedef enum Accrqa_Distance Accrqa_Distance;

#endif
