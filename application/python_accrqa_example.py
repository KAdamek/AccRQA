import numpy as np
import accrqa as rqa
import pandas as pd

tau_values = np.array([1,2], dtype=np.intc)
emb_values = np.array([1,2], dtype=np.intc)
lmin_values = np.array([2,3], dtype=np.intc)
vmin_values = np.array([2,3], dtype=np.intc)
threshold_values = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], dtype=np.float32)
input_data = np.random.rand(10)
input_data = input_data.astype('float32')


print("----------------------------------------------")
print("tau:")
print(tau_values)
print("emb:")
print(emb_values)
print("thresholds:")
print(threshold_values)
print("Input:")
print(input_data)
output_RR = rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type='maximal', tidy_data = False)
print("output RR:\n")
#print(output_RR)
print("----------------------------------------------")

print(" ")
output_RR_pd = rqa.RR(input_data, tau_values, emb_values, threshold_values, distance_type='maximal', tidy_data = True)
print(output_RR_pd);
print(" ")

print("----------------------------------------------")
print("lmin:")
print(lmin_values)
output_DET = rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type='maximal', calculate_ENTR = True, tidy_data = False)
print("output DET:\n")
#print(output_DET)
print("----------------------------------------------")

print(" ")
output_DET_pd = rqa.DET(input_data, tau_values, emb_values, lmin_values, threshold_values, distance_type='maximal', calculate_ENTR = True, tidy_data = True)
print(output_DET_pd);
print(" ")


print("----------------------------------------------")
print("vmin:")
print(vmin_values)
output_LAM = rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type='maximal', calculate_ENTR = True, tidy_data = False)
print("output LAM:\n")
#print(output_LAM)
print("----------------------------------------------")

print(" ")
output_LAM_pd = rqa.LAM(input_data, tau_values, emb_values, vmin_values, threshold_values, distance_type='maximal', calculate_ENTR = True, tidy_data = True)
print(output_LAM_pd);
print(" ")



