#ifndef _ACCRQA_RECURRENTRATE_HPP
#define _ACCRQA_RECURRENTRATE_HPP

#include <vector>
#include <set>

/** \enum accrqa_metrics
 * \brief Contains RQA metrics.
 */
enum class accrqa_Metrics : int {
	empty = 0, //< Nothing is calculated.
	RR,        //< Recurrent rate
	DET,       //< Determinism
	L,         //< Averaged diagonal line length
	LMAX,      //< Maximal diagonal line length
	LAM,       //< Laminarity
	T,         //< Trapping time
	TMAX,      //< Maximum trapping time
	ENTR,      //< Entropy
	TREND,     //< Trend
};

class AccRQA_Config {
public:
	std::vector<double> thresholds;
	std::vector<int> emb;
	std::vector<int> tau;
	std::vector<int> lmin;
	std::vector<int> vmin;
	std::set<accrqa_Metrics> metrics;
};

struct accrqa_Result {
	double RR, DET, L, LMAX, LAM, T, TMAX, ENTR, TREND;
	double threshold;
	int emb, tau, lmin, vmin;
};


/*
	std::set<aa_pipeline::component_option> pipeline_option;
	
	// Check if metric is there
	if (m_pipeline_options.find(opt_set_bandpass_average) != m_pipeline_options.end()) 
	// ADD options
	m_pipeline_options.insert(aa_pipeline::component_option::output_DDTR_normalization);
*/

#endif

