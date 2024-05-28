# See the LICENSE file at the top-level directory of this distribution.

import ctypes
import numpy as np

try:
    import cupy
except ImportError:
    cupy = None

try:
    import pandas as pd
except ImportError:
    pd = None
    
#TODO: separate functions for distance and computation platform
#TODO: improve support for tidy data, preferably directly from C

from . import accrqaError
from . import accrqaLib
from . import accrqaMem

def RR(input_data, tau_values, emb_values, threshold_values, distance_type, comp_platform='nv_gpu', tidy_data=False):
    if tidy_data == True and pd == None:
        raise Exception("Error: Pandas required for tidy data format!")
    
    nTaus = tau_values.shape[0]
    nEmbs = emb_values.shape[0]
    nThresholds = threshold_values.shape[0]
        
    if type(input_data) == np.ndarray:
        rqa_metrics = np.zeros((nTaus, nEmbs, nThresholds), dtype=input_data.dtype);
    else:
        raise TypeError("Unknown array type")

    int_distance_type = 0;
    if distance_type == 'euclidean':
        int_distance_type = 1
    elif distance_type == 'maximal':
        int_distance_type = 2
    else:
        raise TypeError("Unknown distance type")
    
    int_comp_platform = 0;
    if comp_platform == 'nv_gpu':
        int_comp_platform = 1024
    elif comp_platform == 'cpu':
        int_comp_platform = 1
    else:
        raise TypeError("Unknown compute platform")
    
    mem_output = accrqaMem(rqa_metrics)
    mem_input = accrqaMem(input_data)
    mem_tau_values = accrqaMem(tau_values)
    mem_emb_values = accrqaMem(emb_values)
    mem_threshold_values = accrqaMem(threshold_values)
    error_status = accrqaError()
    lib_AccRQA = accrqaLib.handle().py_accrqa_RR
    lib_AccRQA.argtypes = [
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        accrqaError.handle_type()
    ]
    lib_AccRQA(
        mem_output.handle(),
        mem_input.handle(),
        mem_tau_values.handle(),
        mem_emb_values.handle(),
        mem_threshold_values.handle(),
        int_distance_type,
        int_comp_platform,
        error_status.handle()
    )
    error_status.check()
    
    if tidy_data == False:
        return(rqa_metrics)
    if tidy_data == True:
        tmplist = []
        for tau_idx, tau in enumerate(tau_values):
            for emb_idx, emb in enumerate(emb_values):
                for thr_idx, thr in enumerate(threshold_values):
                    RRvalue = rqa_metrics[tau_idx, emb_idx, thr_idx]
                    d = {
                        'Delay' : tau,
                        'Embedding' : emb,
                        'Threshold' : thr,
                        'RR' : RRvalue
                    }
                    tmplist.append(d)
        
        tidy_format_result = pd.DataFrame(tmplist)
        return(tidy_format_result);

def DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type, calculate_ENTR, comp_platform='nv_gpu', tidy_data=False):
    if tidy_data == True and pd == None:
        raise Exception("Error: Pandas required for tidy data format!")
    
    nTaus = tau_values.shape[0]
    nEmbs = emb_values.shape[0]
    nLmins = lmin_values.shape[0]
    nThresholds = threshold_values.shape[0]
    
    if type(input_data) == np.ndarray:
        rqa_metrics = np.zeros((nTaus, nEmbs, nLmins, nThresholds, 5), dtype=input_data.dtype);
    else:
        raise TypeError("Unknown array type")

    int_distance_type = 0;
    if distance_type == 'euclidean':
        int_distance_type = 1
    elif distance_type == 'maximal':
        int_distance_type = 2
    else:
        raise TypeError("Unknown distance type")
    
    int_comp_platform = 0;
    if comp_platform == 'nv_gpu':
        int_comp_platform = 1024
    elif comp_platform == 'cpu':
        int_comp_platform = 1
    else:
        raise TypeError("Unknown compute platform")
    
    int_calc_ENTR = 0;
    if calculate_ENTR == True:
        int_calc_ENTR = 1
    elif calculate_ENTR == False:
        int_calc_ENTR = 0
    else:
        raise TypeError("Invalid value of calculate_ENTR")
    
    mem_output = accrqaMem(rqa_metrics)
    mem_input = accrqaMem(input_data)
    mem_tau_values = accrqaMem(tau_values)
    mem_emb_values = accrqaMem(emb_values)
    mem_lmin_values = accrqaMem(lmin_values)
    mem_threshold_values = accrqaMem(threshold_values)
    error_status = accrqaError()
    lib_AccRQA = accrqaLib.handle().py_accrqa_DET
    lib_AccRQA.argtypes = [
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        accrqaError.handle_type()
    ]
    lib_AccRQA(
        mem_output.handle(),
        mem_input.handle(),
        mem_tau_values.handle(),
        mem_emb_values.handle(),
        mem_lmin_values.handle(),
        mem_threshold_values.handle(),
        int_distance_type,
        int_calc_ENTR,
        int_comp_platform,
        error_status.handle()
    )
    error_status.check()
    
    if tidy_data == False:
        return(rqa_metrics)
    if tidy_data == True:
        tmplist = []
        for tau_idx, tau in enumerate(tau_values):
            for emb_idx, emb in enumerate(emb_values):
                for lmin_idx, lmin in enumerate(lmin_values):
                    for thr_idx, thr in enumerate(threshold_values):
                        DET  = rqa_metrics[tau_idx, emb_idx, lmin_idx, thr_idx, 0]
                        L    = rqa_metrics[tau_idx, emb_idx, lmin_idx, thr_idx, 1]
                        Lmax = rqa_metrics[tau_idx, emb_idx, lmin_idx, thr_idx, 2]
                        ENTR = rqa_metrics[tau_idx, emb_idx, lmin_idx, thr_idx, 3]
                        RR   = rqa_metrics[tau_idx, emb_idx, lmin_idx, thr_idx, 4]
                        d = {
                            'Delay' : tau,
                            'Embedding' : emb,
                            'Lmin' : lmin,
                            'Threshold' : thr,
                            'DET' : DET,
                            'L' : L,
                            'Lmax' : Lmax,
                            'ENTR' : ENTR,
                            'RR' : RR
                        }
                        tmplist.append(d)
        
        tidy_format_result = pd.DataFrame(tmplist)
        return(tidy_format_result);

def LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type, calculate_ENTR, comp_platform='nv_gpu', tidy_data=False):
    if tidy_data == True and pd == None:
        raise Exception("Error: Pandas required for tidy data format!")
    
    nTaus = tau_values.shape[0]
    nEmbs = emb_values.shape[0]
    nVmins = vmin_values.shape[0]
    nThresholds = threshold_values.shape[0]
    
    if type(input_data) == np.ndarray:
        rqa_metrics = np.zeros((nTaus, nEmbs, nVmins, nThresholds, 5), dtype=input_data.dtype);
    else:
        raise TypeError("Unknown array type")

    int_distance_type = 0;
    if distance_type == 'euclidean':
        int_distance_type = 1
    elif distance_type == 'maximal':
        int_distance_type = 2
    else:
        raise TypeError("Unknown distance type")
    
    int_calc_ENTR = 0;
    if calculate_ENTR == True:
        int_calc_ENTR = 1
    elif calculate_ENTR == False:
        int_calc_ENTR = 0
    else:
        raise TypeError("Invalid value of calculate_ENTR")
    
    int_comp_platform = 0;
    if comp_platform == 'nv_gpu':
        int_comp_platform = 1024
    elif comp_platform == 'cpu':
        int_comp_platform = 1
    else:
        raise TypeError("Unknown compute platform")
    
    mem_output = accrqaMem(rqa_metrics)
    mem_input = accrqaMem(input_data)
    mem_tau_values = accrqaMem(tau_values)
    mem_emb_values = accrqaMem(emb_values)
    mem_vmin_values = accrqaMem(vmin_values)
    mem_threshold_values = accrqaMem(threshold_values)
    error_status = accrqaError()
    lib_AccRQA = accrqaLib.handle().py_accrqa_LAM
    lib_AccRQA.argtypes = [
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        accrqaError.handle_type()
    ]
    lib_AccRQA(
        mem_output.handle(),
        mem_input.handle(),
        mem_tau_values.handle(),
        mem_emb_values.handle(),
        mem_vmin_values.handle(),
        mem_threshold_values.handle(),
        int_distance_type,
        int_calc_ENTR,
        int_comp_platform,
        error_status.handle()
    )
    error_status.check()
    
    if tidy_data == False:
        return(rqa_metrics)
    if tidy_data == True:
        tmplist = []
        for tau_idx, tau in enumerate(tau_values):
            for emb_idx, emb in enumerate(emb_values):
                for vmin_idx, vmin in enumerate(vmin_values):
                    for thr_idx, thr in enumerate(threshold_values):
                        LAM   = rqa_metrics[tau_idx, emb_idx, vmin_idx, thr_idx, 0]
                        TT    = rqa_metrics[tau_idx, emb_idx, vmin_idx, thr_idx, 1]
                        TTmax = rqa_metrics[tau_idx, emb_idx, vmin_idx, thr_idx, 2]
                        ENTR  = rqa_metrics[tau_idx, emb_idx, vmin_idx, thr_idx, 3]
                        RR    = rqa_metrics[tau_idx, emb_idx, vmin_idx, thr_idx, 4]
                        d = {
                            'Delay' : tau,
                            'Embedding' : emb,
                            'Vmin' : vmin,
                            'Threshold' : thr,
                            'LAM' : LAM,
                            'TT' : TT,
                            'TTmax' : TTmax,
                            'ENTR' : ENTR,
                            'RR' : RR
                        }
                        tmplist.append(d)
        
        tidy_format_result = pd.DataFrame(tmplist)
        return(tidy_format_result);
