import accrqa as rqa
import numpy as np
import pandas as pd
import pytest

def create_data():
    tau_values = np.array([1,2], dtype=np.intc)
    emb_values = np.array([1,2], dtype=np.intc)
    lmin_values = np.array([2,3], dtype=np.intc)
    vmin_values = np.array([2,3], dtype=np.intc)
    threshold_values = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], dtype=np.float32)
    input_data = np.random.rand(10)
    input_data = input_data.astype('float32')
    return(tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data)

def test_unhappy_RR_tau():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type=distance_type_to_use, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_RR_emb():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    emb_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type=distance_type_to_use, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_RR_threshold():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    threshold_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type=distance_type_to_use, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_RR_input():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    input_data = "Hello!"
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type=distance_type_to_use, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_RR_distance_type():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = "maximal"
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type=distance_type_to_use, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_RR_comp_platform():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = "nv_gpu"
    with pytest.raises(TypeError):
        rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type=distance_type_to_use, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_RR_tidy_data():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type=distance_type_to_use, comp_platform = computational_platform_to_use, tidy_data = "Yes")

#----------------------------


def test_unhappy_DET_tau():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_DET_emb():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    emb_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_DET_threshold():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    threshold_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_DET_lmin():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    lmin_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_DET_input():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    input_data = "Hello!"
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_DET_distance_type():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = "maximal"
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_DET_comp_platform():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = "nv_gpu"
    with pytest.raises(TypeError):
        rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_DET_tidy_data():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = "Yes")

def test_unhappy_DET_ENTR():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = "Yes", comp_platform = computational_platform_to_use, tidy_data = False)

#----------------------------

def test_unhappy_LAM_tau():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_LAM_emb():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    emb_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_LAM_threshold():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    threshold_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_LAM_vmin():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    vmin_values = np.array([], dtype=np.intc)
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_LAM_input():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    input_data = "Hello!"
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_LAM_distance_type():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = "maximal"
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_LAM_comp_platform():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = "nv_gpu"
    with pytest.raises(TypeError):
        rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = False)

def test_unhappy_LAM_tidy_data():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = "Yes")

def test_unhappy_LAM_ENTR():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    distance_type_to_use = rqa.accrqaDistance("maximal")
    computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    with pytest.raises(TypeError):
        rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=distance_type_to_use, calculate_ENTR = "Yes", comp_platform = computational_platform_to_use, tidy_data = False)









