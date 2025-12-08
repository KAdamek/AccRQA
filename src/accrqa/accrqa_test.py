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

# ----------------------------

def test_unhappy_RP_tau():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    # tau must be > 0
    tau = 0
    emb = int(emb_values[0])
    threshold = float(threshold_values[0])
    distance_type_to_use = rqa.accrqaDistance("maximal")
    with pytest.raises(TypeError):
        rqa.RP(input_data, tau, emb, threshold, distance_type_to_use)

def test_unhappy_RP_emb():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = int(tau_values[0])
    # emb must be > 0
    emb = 0
    threshold = float(threshold_values[0])
    distance_type_to_use = rqa.accrqaDistance("maximal")
    with pytest.raises(TypeError):
        rqa.RP(input_data, tau, emb, threshold, distance_type_to_use)

def test_unhappy_RP_input():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    # input_data must be a numpy ndarray
    input_data = "Hello!"
    tau = int(tau_values[0])
    emb = int(emb_values[0])
    threshold = float(threshold_values[0])
    distance_type_to_use = rqa.accrqaDistance("maximal")
    with pytest.raises(TypeError):
        rqa.RP(input_data, tau, emb, threshold, distance_type_to_use)

def test_unhappy_RP_distance_type():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = int(tau_values[0])
    emb = int(emb_values[0])
    threshold = float(threshold_values[0])
    # distance_type must be an instance of accrqaDistance
    distance_type_to_use = "maximal"
    with pytest.raises(TypeError):
        rqa.RP(input_data, tau, emb, threshold, distance_type_to_use)

# ----------------------------

def test_unhappy_RR_target_input():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    input_data = "Hello!"
    tau = int(tau_values[0])
    emb = int(emb_values[0])
    target_RR = 0.5
    distance_type_to_use = rqa.accrqaDistance("maximal")
    comp_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")

    with pytest.raises(TypeError):
        rqa.RR_target(
            input_data,
            tau,
            emb,
            target_RR,
            distance_type_to_use,
            epsilon=0.01,
            comp_platform=comp_platform_to_use,
            max_iter=20,
            threshold_min=0.0,
            threshold_max=10.0,
        )


def test_unhappy_RR_target_tau():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = 0  # must be > 0
    emb = int(emb_values[0])
    target_RR = 0.5
    distance_type_to_use = rqa.accrqaDistance("maximal")
    comp_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")

    with pytest.raises(TypeError):
        rqa.RR_target(
            input_data,
            tau,
            emb,
            target_RR,
            distance_type_to_use,
            epsilon=0.01,
            comp_platform=comp_platform_to_use,
            max_iter=20,
            threshold_min=0.0,
            threshold_max=10.0,
        )


def test_unhappy_RR_target_emb():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = int(tau_values[0])
    emb = 0  # must be > 0
    target_RR = 0.5
    distance_type_to_use = rqa.accrqaDistance("maximal")
    comp_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")

    with pytest.raises(TypeError):
        rqa.RR_target(
            input_data,
            tau,
            emb,
            target_RR,
            distance_type_to_use,
            epsilon=0.01,
            comp_platform=comp_platform_to_use,
            max_iter=20,
            threshold_min=0.0,
            threshold_max=10.0,
        )


def test_unhappy_RR_target_target_RR():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = int(tau_values[0])
    emb = int(emb_values[0])
    target_RR = 1.1  # must be between 0 and 1
    distance_type_to_use = rqa.accrqaDistance("maximal")
    comp_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")

    with pytest.raises(TypeError):
        rqa.RR_target(
            input_data,
            tau,
            emb,
            target_RR,
            distance_type_to_use,
            epsilon=0.01,
            comp_platform=comp_platform_to_use,
            max_iter=20,
            threshold_min=0.0,
            threshold_max=10.0,
        )


def test_unhappy_RR_target_distance_type():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = int(tau_values[0])
    emb = int(emb_values[0])
    target_RR = 0.5
    distance_type_to_use = "maximal"  # must be accrqaDistance instance
    comp_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")

    with pytest.raises(TypeError):
        rqa.RR_target(
            input_data,
            tau,
            emb,
            target_RR,
            distance_type_to_use,
            epsilon=0.01,
            comp_platform=comp_platform_to_use,
            max_iter=20,
            threshold_min=0.0,
            threshold_max=10.0,
        )


def test_unhappy_RR_target_comp_platform():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = int(tau_values[0])
    emb = int(emb_values[0])
    target_RR = 0.5
    distance_type_to_use = rqa.accrqaDistance("maximal")
    comp_platform_to_use = "nv_gpu"  # must be accrqaCompPlatform instance

    with pytest.raises(TypeError):
        rqa.RR_target(
            input_data,
            tau,
            emb,
            target_RR,
            distance_type_to_use,
            epsilon=0.01,
            comp_platform=comp_platform_to_use,
            max_iter=20,
            threshold_min=0.0,
            threshold_max=10.0,
        )


def test_unhappy_RR_target_epsilon():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = int(tau_values[0])
    emb = int(emb_values[0])
    target_RR = 0.5
    distance_type_to_use = rqa.accrqaDistance("maximal")
    comp_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    epsilon = 0.0  # must be > 0

    with pytest.raises(TypeError):
        rqa.RR_target(
            input_data,
            tau,
            emb,
            target_RR,
            distance_type_to_use,
            epsilon=epsilon,
            comp_platform=comp_platform_to_use,
            max_iter=20,
            threshold_min=0.0,
            threshold_max=10.0,
        )


def test_unhappy_RR_target_max_iter():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = int(tau_values[0])
    emb = int(emb_values[0])
    target_RR = 0.5
    distance_type_to_use = rqa.accrqaDistance("maximal")
    comp_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    max_iter = 0  # must be > 0

    with pytest.raises(TypeError):
        rqa.RR_target(
            input_data,
            tau,
            emb,
            target_RR,
            distance_type_to_use,
            epsilon=0.01,
            comp_platform=comp_platform_to_use,
            max_iter=max_iter,
            threshold_min=0.0,
            threshold_max=10.0,
        )


def test_unhappy_RR_target_threshold_range():
    tau_values, emb_values, lmin_values, vmin_values, threshold_values, input_data = create_data()
    tau = int(tau_values[0])
    emb = int(emb_values[0])
    target_RR = 0.5
    distance_type_to_use = rqa.accrqaDistance("maximal")
    comp_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
    threshold_min = 5.0
    threshold_max = 1.0  # threshold_min must be smaller than threshold_max

    with pytest.raises(TypeError):
        rqa.RR_target(
            input_data,
            tau,
            emb,
            target_RR,
            distance_type_to_use,
            epsilon=0.01,
            comp_platform=comp_platform_to_use,
            max_iter=20,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )










