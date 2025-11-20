# See the LICENSE file at the top-level directory of this distribution.

import sys
import ctypes
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional
from typing import Union

try:
    import pandas as pd
except ImportError:
    pd = None
    
#TODO: improve support for tidy data, preferably directly from C

from . import accrqaError
from . import accrqaLib
from . import accrqaMem
from . import accrqaDistance
from . import accrqaCompPlatform

def RP(input_data: NDArray, tau: int, emb: int, threshold: float, distance_type: accrqaDistance) -> NDArray:
    """
    Calculates recurrence plot from supplied time-series.
    https://en.wikipedia.org/wiki/Recurrence_quantification_analysis
    
    Args:
        input_data: The input time-series.
        tau: Integer value of delay.
        emb: Integer value of embedding.
        threshold: Floating point value of threshold.
        distance_type: Norm used to calculate distance. Must be instance of :func:`~accrqa.accrqaDistance`.

    Returns:
        A numpy NDArray containing of RP values.

    Raises:
        TypeError: If number of delays, embedding or thresholds is zero length.
        TypeError: If input_data is not numpy.ndarray.
        TypeError: If wrong type of the distance to the line of identity is selected.
        TypeError: If wrong computational platform is selected.
        RuntimeError: If AccRQA library encounters a problem.
    """
    if tau <= 0 or emb <= 0:
        raise TypeError("Delay and embedding must be greater than zero.")
    
    input_size = input_data.shape[0]
    corrected_size = input_size - (emb - 1)*tau;
    if type(input_data) == np.ndarray:
        rp_output = np.zeros((corrected_size, corrected_size), dtype=np.int8);
    else:
        raise TypeError("Unknown array type of the input_data")
    
    # Checking distance validity
    if not type(distance_type) is accrqaDistance:
        raise TypeError("Distance must be an instance of accrqaDistance")
    int_distance_type = 0
    int_distance_type = distance_type.get_distance_id()
    
    mem_output = accrqaMem(rp_output)
    mem_input = accrqaMem(input_data)
    error_status = accrqaError()
    lib_AccRQA = accrqaLib.handle().py_accrqa_RP
    lib_AccRQA.argtypes = [
        accrqaMem.handle_type(),
        accrqaMem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        accrqaError.handle_type()
    ]
    lib_AccRQA(
        mem_output.handle(),
        mem_input.handle(),
        tau,
        emb,
        threshold,
        int_distance_type,
        error_status.handle()
    )
    error_status.check()
    
    return(rp_output)

def RR(input_data: NDArray, tau_values: ArrayLike, emb_values: ArrayLike, threshold_values: ArrayLike, distance_type: accrqaDistance, comp_platform: Optional[accrqaCompPlatform] = accrqaCompPlatform("nv_gpu"), tidy_data: Optional[bool] = True) -> Union[NDArray, pd.DataFrame]:
    """
    Calculates RR measure from supplied time-series.
    https://en.wikipedia.org/wiki/Recurrence_quantification_analysis
    
    Args:
        input_data: The input time-series.
        tau_values: Array of delays.
        emb_values: Array of embedding values.
        threshold_values: Array of threshold values.
        distance_type: Norm used to calculate distance. Must be instance of :func:`~accrqa.accrqaDistance`.
        comp_platform: [Optional] Computational platform to be used. Default is cpu. Must be instance of :func:`~accrqa.accrqaCompPlatform`.
        tidy_data: [Optional] Output data in tidy data format. Requires pandas.

    Returns:
        A numpy NDArray containing of RR values with dimensions [number of delays, 
        number of embeddings, number of thresholds].

    Raises:
        TypeError: If number of delays, embedding or thresholds is zero length.
        TypeError: If input_data is not numpy.ndarray.
        TypeError: If wrong type of the distance to the line of identity is selected.
        TypeError: If wrong computational platform is selected.
        RuntimeError: If AccRQA library encounters a problem.
    """
    pandas_detected = 'datetime' in sys.modules
    if tidy_data == True and pandas_detected == False:
        raise Exception("Error: Pandas required for tidy data format!")
    
    if not type(tidy_data) == bool:
        raise TypeError("tidy_data must be bool (False or True)")
    
    nTaus = tau_values.shape[0]
    nEmbs = emb_values.shape[0]
    nThresholds = threshold_values.shape[0]
    
    if nTaus <= 0 or nEmbs <= 0 or nThresholds <= 0:
        raise TypeError("Number of delays, embedding or thresholds must be greater than zero.")
        
    if type(input_data) == np.ndarray:
        rqa_metrics = np.zeros((nTaus, nEmbs, nThresholds), dtype=input_data.dtype);
    else:
        raise TypeError("Unknown array type of the input_data")
    
    # Checking distance validity
    if not type(distance_type) is accrqaDistance:
        raise TypeError("Distance must be an instance of accrqaDistance")
    int_distance_type = 0
    int_distance_type = distance_type.get_distance_id()
    
    # Checking computational platform validity
    if not type(comp_platform) is accrqaCompPlatform:
        raise TypeError("Computational platform must be an instance of accrqaCompPlatform")
    int_comp_platform = 0
    int_comp_platform = comp_platform.get_platform_id()
    
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

def DET(input_data: NDArray, tau_values: ArrayLike, emb_values: ArrayLike, lmin_values: ArrayLike, threshold_values: ArrayLike, distance_type: accrqaDistance, calculate_ENTR: Optional[bool] = True, comp_platform: Optional[accrqaCompPlatform] = accrqaCompPlatform("nv_gpu"), tidy_data: Optional[bool] = True) -> Union[NDArray, pd.DataFrame]:
    """
    Calculates DET, L, Lmax, ENTR and RR measures from supplied time-series.
    https://en.wikipedia.org/wiki/Recurrence_quantification_analysis
    
    Args:
        input_data: The input time-series.
        tau_values: Array of delays.
        emb_values: Array of embedding values.
        lmin_values: Array of minimal lengths.
        threshold_values: Array of threshold values.
        distance_type: Norm used to calculate distance. Must be instance of :func:`~accrqa.accrqaDistance`.
        calculate_ENTR: [Optional] Enable calculation of Lmax and ENTR. Default True.
        comp_platform: [Optional] Computational platform to be used. Default is cpu. Must be instance of :func:`~accrqa.accrqaCompPlatform`.
        tidy_data: [Optional] Output data in tidy data format. Requires pandas.

    Returns:
        A numpy NDArray containing of RR values with dimensions [number of delays, 
        number of embeddings, number of thresholds].

    Raises:
        TypeError: If number of delays, embedding, minimal lengths or thresholds is zero length.
        TypeError: If input_data is not numpy.ndarray.
        TypeError: If wrong type of the distance to the line of identity is selected.
        TypeError: If wrong computational platform is selected.
        RuntimeError: If AccRQA library encounters a problem.
    """
    
    pandas_detected = 'datetime' in sys.modules
    if tidy_data == True and pandas_detected == False:
        raise Exception("Error: Pandas required for tidy data format!")
    
    if not type(tidy_data) == bool:
        raise TypeError("tidy_data must be bool (False or True)")
    
    if not type(calculate_ENTR) == bool:
        raise TypeError("calculate_ENTR must be bool (False or True)")
    
    nTaus = tau_values.shape[0]
    nEmbs = emb_values.shape[0]
    nLmins = lmin_values.shape[0]
    nThresholds = threshold_values.shape[0]
    
    if nTaus <= 0 or nEmbs <= 0 or nLmins <= 0 or nThresholds <= 0:
        raise TypeError("Number of delays, embedding, minimal lengths or thresholds must be greater than zero.")
    
    if type(input_data) == np.ndarray:
        rqa_metrics = np.zeros((nTaus, nEmbs, nLmins, nThresholds, 5), dtype=input_data.dtype);
    else:
        raise TypeError("Unknown array type")
    
    # Checking distance validity
    if not type(distance_type) is accrqaDistance:
        raise TypeError("distance must be an instance of accrqaDistance")
    int_distance_type = 0
    int_distance_type = distance_type.get_distance_id()
    
    # Checking computational platform validity
    if not type(comp_platform) is accrqaCompPlatform:
        raise TypeError("Computational platform must be an instance of accrqaCompPlatform")
    int_comp_platform = 0
    int_comp_platform = comp_platform.get_platform_id()
    
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

def LAM(input_data: NDArray, tau_values: ArrayLike, emb_values: ArrayLike, vmin_values: ArrayLike, threshold_values: ArrayLike, distance_type: accrqaDistance, calculate_ENTR: Optional[bool] = True, comp_platform: Optional[accrqaCompPlatform] = accrqaCompPlatform("nv_gpu"), tidy_data: Optional[bool] = True) -> Union[NDArray, pd.DataFrame]:
    """
    Calculates DET, L, Lmax, ENTR and RR measures from supplied time-series.
    https://en.wikipedia.org/wiki/Recurrence_quantification_analysis
    
    Args:
        input_data: The input time-series.
        tau_values: Array of delays.
        emb_values: Array of embedding values.
        vmin_values: Array of minimal lengths.
        threshold_values: Array of threshold values.
        distance_type: Norm used to calculate distance. Must be instance of :func:`~accrqa.accrqaDistance`.
        calculate_ENTR: [Optional] Enable calculation of Vmax and ENTR. Default True.
        comp_platform: [Optional] Computational platform to be used. Default is cpu. Must be instance of :func:`~accrqa.accrqaCompPlatform`.
        tidy_data: [Optional] Output data in tidy data format. Requires pandas.

    Returns:
        A numpy NDArray containing of RR values with dimensions [number of delays, 
        number of embeddings, number of thresholds].

    Raises:
        TypeError: If number of delays, embedding, minimal lengths or thresholds is zero length.
        TypeError: If input_data is not numpy.ndarray.
        TypeError: If wrong type of the distance to the line of identity is selected.
        TypeError: If wrong computational platform is selected.
        RuntimeError: If AccRQA library encounters a problem.
    """
    
    pandas_detected = 'datetime' in sys.modules
    if tidy_data == True and pandas_detected == False:
        raise Exception("Error: Pandas required for tidy data format!")
    
    if not type(tidy_data) == bool:
        raise TypeError("tidy_data must be bool (False or True)")
    
    if not type(calculate_ENTR) == bool:
        raise TypeError("calculate_ENTR must be bool (False or True)")
    
    nTaus = tau_values.shape[0]
    nEmbs = emb_values.shape[0]
    nVmins = vmin_values.shape[0]
    nThresholds = threshold_values.shape[0]
    
    if nTaus <= 0 or nEmbs <= 0 or nVmins <= 0 or nThresholds <= 0:
        raise TypeError("Number of delays, embedding, minimal lengths or thresholds must be greater than zero.")
    
    if type(input_data) == np.ndarray:
        rqa_metrics = np.zeros((nTaus, nEmbs, nVmins, nThresholds, 5), dtype=input_data.dtype);
    else:
        raise TypeError("Unknown array type")
    
    # Checking distance validity
    if not type(distance_type) is accrqaDistance:
        raise TypeError("distance must be an instance of accrqaDistance")
    int_distance_type = 0
    int_distance_type = distance_type.get_distance_id()
    
    # Checking computational platform validity
    if not type(comp_platform) is accrqaCompPlatform:
        raise TypeError("Computational platform must be an instance of accrqaCompPlatform")
    int_comp_platform = 0
    int_comp_platform = comp_platform.get_platform_id()
    
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
