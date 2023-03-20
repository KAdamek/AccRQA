#ifndef _ACCRQA_LENGTHHISTOGRAM_HPP
#define _ACCRQA_LENGTHHISTOGRAM_HPP

#include "AccRQA_utitlities_mem.hpp"
#include "AccRQA_utitlities_error.hpp"

void accrqa_RR(Mem *mem_output, Mem *mem_input, Mem *mem_tau_values, Mem *mem_emb_values, Mem *mem_threshold_values, int distance_type, Accrqa_Error *error);

#endif