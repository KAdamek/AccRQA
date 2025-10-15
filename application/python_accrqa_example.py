import numpy as np
import accrqa as rqa
import pandas as pd

tau_values = np.array([1,2], dtype=np.intc)
emb_values = np.array([1,2], dtype=np.intc)
lmin_values = np.array([2,3], dtype=np.intc)
vmin_values = np.array([2,3], dtype=np.intc)
threshold_values = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], dtype=np.float32)
input_data = np.random.rand(100)
input_data = input_data.astype('float32')


print("---- Initial parameters ----")
print("tau:")
print(tau_values)
print("emb:")
print(emb_values)
print("thresholds:")
print(threshold_values)
print("lmin:")
print(lmin_values)
print("vmin:")
print(vmin_values)
print("Input:")
print(input_data)
print("----------------------------")

print(" ")

print("----------- RR -------------")
print("Output as data cube if we set tidy_data to False:")
output_RR = rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type=rqa.accrqaDistance("maximal"), comp_platform = rqa.accrqaCompPlatform("nv_gpu"), tidy_data = False)
print(output_RR)
print(" ")
print("Output in tidy-data format using Pandas:")
distance_type_to_use = rqa.accrqaDistance("maximal")
output_RR_pd = rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type=distance_type_to_use, comp_platform = rqa.accrqaCompPlatform("nv_gpu"), tidy_data = True)
print(output_RR_pd);
print("----------------------------")

print(" ")

print("----------- DET ------------")
computational_platform_to_use = rqa.accrqaCompPlatform("nv_gpu")
output_DET_pd = rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type=rqa.accrqaDistance("maximal"), calculate_ENTR = True, comp_platform = computational_platform_to_use, tidy_data = True)
print("DET tidy-data output:")
print(output_DET_pd);
print("----------------------------")

print(" ")

print("----------- LAM ------------")
output_LAM_pd = rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type=rqa.accrqaDistance("maximal"), calculate_ENTR = True, comp_platform = rqa.accrqaCompPlatform("nv_gpu"), tidy_data = True)
print("LAM tidy-data output:")
print(output_LAM_pd);
print("----------------------------")



