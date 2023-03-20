# See the LICENSE file at the top-level directory of this distribution.

import ctypes
import numpy as np
try:
    import cupy
except ImportError:
    cupy = None

#import numpy.ctypeslib as ctl
from accrqa_error import Error
from accrqa_lib import Lib
from accrqa_mem import Mem

def accrqa_RR(input_data, tau_values, emb_values, threshold_values, distance_type):
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
    
    mem_output = Mem(rqa_metrics)
    mem_input = Mem(input_data)
    mem_tau_values = Mem(tau_values)
    mem_emb_values = Mem(emb_values)
    mem_threshold_values = Mem(threshold_values)
    error_status = Error()
    lib_AccRQA = Lib.handle().py_accrqa_RR
    lib_AccRQA.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        Error.handle_type()
    ]
    lib_AccRQA(
        mem_output.handle(),
        mem_input.handle(),
        mem_tau_values.handle(),
        mem_emb_values.handle(),
        mem_threshold_values.handle(),
        int_distance_type,
        error_status.handle()
    )
    error_status.check()
    return(rqa_metrics)

def accrqa_DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type, calculate_ENTR):
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
    
    int_calc_ENTR = 0;
    if calculate_ENTR == 'true':
        int_calc_ENTR = 1
    elif calculate_ENTR == 'false':
        int_calc_ENTR = 0
    else:
        raise TypeError("Invalid value of calculate_ENTR")
    
    mem_output = Mem(rqa_metrics)
    mem_input = Mem(input_data)
    mem_tau_values = Mem(tau_values)
    mem_emb_values = Mem(emb_values)
    mem_lmin_values = Mem(lmin_values)
    mem_threshold_values = Mem(threshold_values)
    error_status = Error()
    lib_AccRQA = Lib.handle().py_accrqa_DET
    lib_AccRQA.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        Error.handle_type()
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
        error_status.handle()
    )
    error_status.check()
    return(rqa_metrics)

def accrqa_LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type, calculate_ENTR):
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
    if calculate_ENTR == 'true':
        int_calc_ENTR = 1
    elif calculate_ENTR == 'false':
        int_calc_ENTR = 0
    else:
        raise TypeError("Invalid value of calculate_ENTR")
    
    mem_output = Mem(rqa_metrics)
    mem_input = Mem(input_data)
    mem_tau_values = Mem(tau_values)
    mem_emb_values = Mem(emb_values)
    mem_vmin_values = Mem(vmin_values)
    mem_threshold_values = Mem(threshold_values)
    error_status = Error()
    lib_AccRQA = Lib.handle().py_accrqa_LAM
    lib_AccRQA.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        Error.handle_type()
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
        error_status.handle()
    )
    error_status.check()
    return(rqa_metrics)
