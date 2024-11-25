#ifndef _ACCRQA_FUNCTIONS_HPP
#define _ACCRQA_FUNCTIONS_HPP

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <set>

#include "AccRQA_utilities_error.hpp"
#include "AccRQA_utilities_comp_platform.hpp"
#include "AccRQA_utilities_distance.hpp"

/**
 * @defgroup rqa_metrics
 * @{
 */

/**
 * @brief Prints error messages associated with error status.
 *
 * @param error Error status to print.
 */
void accrqa_print_error(Accrqa_Error *error);

//==========================================================
//========================= LAM ============================
//==========================================================

/**
 * @brief Calculates LAM, TT, TTmax, ENTR and RR RQA metrics from supplied time-series. (float)
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 * - @p output is 3D data cube containing RR values, with shape:
 *   - [ @p nTaus, @p nEmbs, @p nLmins, @p nThresholds , 5 ].
 *
 * - @p input_data is 1D and real-valued, with shape:
 *   - [ @p data_size ]
 *
 * - @p tau_values is 1D and integer-valued, with shape:
 *   - [ @p nTaus ]
 *
 * - @p emb_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ] 
 *
 * - @p lmin_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ]
 *
 * - @p threshold_values is 1D and real-valued, with shape:
 *   - [ @p nThresholds ]
 *
 * @param output Multi-dimensional data cube containing RR values.
 * @param input_data Real-valued array of input time-series samples.
 * @param data_size Number of samples (float) of the time-series.
 * @param tau_values Integer array of delay values.
 * @param nTaus Number of delays.
 * @param emb_values Integer array of embedding values.
 * @param nEmbs Number of embeddings.
 * @param vmin_values Integer array of  minimal lengths values.
 * @param nVmins Number of  minimal lengths.
 * @param threshold_values Real-valued array (float) of threshold values.
 * @param nThresholds Number of threshold values.
 * @param distance_type Distance formula used in calculation of distance to the line of identity.
 * @param calc_ENTR Turns calculation of ENTR on (1) and off (0).
 * @param comp_platform Compute platform to use.
 * @param error Error status.
 */
void accrqa_LAM(
	float *output, 
	float *input_data, size_t data_size, 
	int *tau_values, int nTaus, 
	int *emb_values, int nEmbs, 
	int *vmin_values, int nVmins, 
	float *threshold_values, int nThresholds, 
	Accrqa_Distance distance_type, int calc_ENTR, Accrqa_CompPlatform comp_platform, 
	Accrqa_Error *error
);

/**
 * @brief Calculates LAM, TT, TTmax, ENTR and RR RQA metrics from supplied time-series. (double)
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 * - @p output is 3D data cube containing RR values, with shape:
 *   - [ @p nTaus, @p nEmbs, @p nLmins, @p nThresholds , 5 ].
 *
 * - @p input_data is 1D and real-valued, with shape:
 *   - [ @p data_size ]
 *
 * - @p tau_values is 1D and integer-valued, with shape:
 *   - [ @p nTaus ]
 *
 * - @p emb_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ] 
 *
 * - @p lmin_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ]
 *
 * - @p threshold_values is 1D and real-valued, with shape:
 *   - [ @p nThresholds ]
 *
 * @param output Multi-dimensional data cube containing RR values.
 * @param input_data Real-valued array of input time-series samples.
 * @param data_size Number of samples (double) of the time-series.
 * @param tau_values Integer array of delay values.
 * @param nTaus Number of delays.
 * @param emb_values Integer array of embedding values.
 * @param nEmbs Number of embeddings.
 * @param vmin_values Integer array of  minimal lengths values.
 * @param nVmins Number of  minimal lengths.
 * @param threshold_values Real-valued array (double) of threshold values.
 * @param nThresholds Number of threshold values.
 * @param distance_type Distance formula used in calculation of distance to the line of identity.
 * @param calc_ENTR Turns calculation of ENTR on (1) and off (0).
 * @param comp_platform Compute platform to use.
 * @param error Error status.
 */
void accrqa_LAM(
	double *output, 
	double *input_data, size_t data_size, 
	int *tau_values, int nTaus, 
	int *emb_values, int nEmbs, 
	int *vmin_values, int nVmins, 
	double *threshold_values, int nThresholds, 
	Accrqa_Distance distance_type, int calc_ENTR, Accrqa_CompPlatform comp_platform, 
	Accrqa_Error *error
);

/**
 * @brief Calculates size of LAM output array in number of elements.
 *
 * @param nTaus Number of delays.
 * @param nEmbs Number of embeddings.
 * @param nVmins Number of minimal lengths.
 * @param nThresholds Number of threshold values.
 */
int accrqa_LAM_output_size_in_elements(
	int nTaus, int nEmbs, int nVmins, int nThresholds
);


//==========================================================
//========================= DET ============================
//==========================================================

/**
 * @brief Calculates DET, L, Lmax, ENTR and RR RQA metrics from supplied time-series. (float)
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 * - @p output is 3D data cube containing RR values, with shape:
 *   - [ @p nTaus, @p nEmbs, @p nLmins, @p nThresholds , 5 ].
 *
 * - @p input_data is 1D and real-valued, with shape:
 *   - [ @p data_size ]
 *
 * - @p tau_values is 1D and integer-valued, with shape:
 *   - [ @p nTaus ]
 *
 * - @p emb_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ] 
 *
 * - @p lmin_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ]
 *
 * - @p threshold_values is 1D and real-valued, with shape:
 *   - [ @p nThresholds ]
 *
 * @param output Multi-dimensional data cube containing RR values.
 * @param input_data Real-valued array of input time-series samples.
 * @param data_size Number of samples (float) of the time-series.
 * @param tau_values Integer array of delay values.
 * @param nTaus Number of delays.
 * @param emb_values Integer array of embedding values.
 * @param nEmbs Number of embeddings.
 * @param lmin_values Integer array of  minimal lengths values.
 * @param nLmins Number of  minimal lengths.
 * @param threshold_values Real-valued array (float) of threshold values.
 * @param nThresholds Number of threshold values.
 * @param distance_type Distance formula used in calculation of distance to the line of identity.
 * @param calc_ENTR Turns calculation of ENTR on (1) and off (0).
 * @param comp_platform Compute platform to use.
 * @param error Error status.
 */
void accrqa_DET(
	float *output, 
	float *input_data, size_t data_size, 
	int *tau_values, int nTaus, 
	int *emb_values, int nEmbs, 
	int *lmin_values, int nLmins, 
	float *threshold_values, int nThresholds, 
	Accrqa_Distance distance_type, int calc_ENTR, Accrqa_CompPlatform comp_platform, 
	Accrqa_Error *error
);

/**
 * @brief Calculates DET, L, Lmax, ENTR and RR RQA metrics from supplied time-series. (double)
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 * - @p output is 3D data cube containing RR values, with shape:
 *   - [ @p nTaus, @p nEmbs, @p nLmins, @p nThresholds , 5 ].
 *
 * - @p input_data is 1D and real-valued, with shape:
 *   - [ @p data_size ]
 *
 * - @p tau_values is 1D and integer-valued, with shape:
 *   - [ @p nTaus ]
 *
 * - @p emb_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ] 
 *
 * - @p lmin_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ]
 *
 * - @p threshold_values is 1D and real-valued, with shape:
 *   - [ @p nThresholds ]
 *
 * @param output Multi-dimensional data cube containing RR values.
 * @param input_data Real-valued array of input time-series samples.
 * @param data_size Number of samples (double) of the time-series.
 * @param tau_values Integer array of delay values.
 * @param nTaus Number of delays.
 * @param emb_values Integer array of embedding values.
 * @param nEmbs Number of embeddings.
 * @param lmin_values Integer array of  minimal lengths values.
 * @param nLmins Number of  minimal lengths.
 * @param threshold_values Real-valued array (double) of threshold values.
 * @param nThresholds Number of threshold values.
 * @param distance_type Distance formula used in calculation of distance to the line of identity.
 * @param calc_ENTR Turns calculation of ENTR on (1) and off (0).
 * @param comp_platform Compute platform to use.
 * @param error Error status.
 */
void accrqa_DET(
	double *output, 
	double *input_data, size_t data_size, 
	int *tau_values, int nTaus, 
	int *emb_values, int nEmbs, 
	int *lmin_values, int nLmins, 
	double *threshold_values, int nThresholds, 
	Accrqa_Distance distance_type, int calc_ENTR, Accrqa_CompPlatform comp_platform,
	Accrqa_Error *error
);

/**
 * @brief Calculates size of DET output array in number of elements.
 *
 * @param nTaus Number of delays.
 * @param nEmbs Number of embeddings.
 * @param nLmins Number of minimal lengths.
 * @param nThresholds Number of threshold values.
 */
int accrqa_DET_output_size_in_elements(
	int nTaus, int nEmbs, int nLmins, int nThresholds
);


//==========================================================
//========================== RR ============================
//==========================================================

/**
 * @brief Calculates RR RQA metric from supplied time-series. (float)
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 * - @p output is 3D data cube containing RR values, with shape:
 *   - [ @p nTaus, @p nEmbs, @p nThresholds ].
 *
 * - @p input_data is 1D and real-valued, with shape:
 *   - [ @p data_size ]
 *
 * - @p tau_values is 1D and integer-valued, with shape:
 *   - [ @p nTaus ]
 *
 * - @p emb_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ]
 *
 * - @p threshold_values is 1D and real-valued, with shape:
 *   - [ @p nThresholds ]
 *
 * @param output Multi-dimensional data cube containing RR values.
 * @param input_data Real-valued array of input time-series samples.
 * @param data_size Number of samples (float) of the time-series.
 * @param tau_values Integer array of delay values.
 * @param nTaus Number of delays.
 * @param emb_values Integer array of embedding values.
 * @param nEmbs Number of embeddings.
 * @param threshold_values Real-valued array (float) of threshold values.
 * @param nThresholds Number of threshold values.
 * @param distance_type Distance formula used in calculation of distance to the line of identity.
 * @param comp_platform Compute platform to use.
 * @param error Error status.
 */
void accrqa_RR(
	float *output, 
	float *input_data, size_t data_size, 
	int *tau_values, int nTaus, 
	int *emb_values, int nEmbs, 
	float *threshold_values, int nThresholds, 
	Accrqa_Distance distance_type, Accrqa_CompPlatform comp_platform, 
	Accrqa_Error *error
);

/**
 * @brief Calculates RR RQA metric from supplied time-series. (double)
 *
 * Array dimensions are as follows, from slowest to fastest varying:
 * - @p output is 3D data cube containing RR values, with shape:
 *   - [ @p nTaus, @p nEmbs, @p nThresholds ].
 *
 * - @p input_data is 1D and real-valued, with shape:
 *   - [ @p data_size ]
 *
 * - @p tau_values is 1D and integer-valued, with shape:
 *   - [ @p nTaus ]
 *
 * - @p emb_values is 1D and integer-valued, with shape:
 *   - [ @p nEmbs ]
 *
 * - @p threshold_values is 1D and real-valued, with shape:
 *   - [ @p nThresholds ]
 *
 * @param output Multi-dimensional data cube containing RR values.
 * @param input_data Real-valued array of input time-series samples.
 * @param data_size Number of samples (double) of the time-series.
 * @param tau_values Integer array of delay values.
 * @param nTaus Number of delays.
 * @param emb_values Integer array of embedding values.
 * @param nEmbs Number of embeddings.
 * @param threshold_values Real-valued array (double) of threshold values.
 * @param nThresholds Number of threshold values.
 * @param distance_type Distance formula used in calculation of distance to the line of identity.
 * @param comp_platform Compute platform to use.
 * @param error Error status.
 */
void accrqa_RR(
	double *output, 
	double *input_data, size_t data_size, 
	int *tau_values, int nTaus, 
	int *emb_values, int nEmbs, 
	double *threshold_values, int nThresholds, 
	Accrqa_Distance distance_type, Accrqa_CompPlatform comp_platform,
	Accrqa_Error *error
);

/**
 * @brief Calculates size of RR output array in number of elements.
 *
 * @param nTaus Number of delays.
 * @param nEmbs Number of embeddings.
 * @param nThresholds Number of threshold values.
 */
int accrqa_RR_output_size_in_elements(
	int nTaus, int nEmbs, int nThresholds
);

/** @} */ /* End group rqa_metrics. */
#endif
