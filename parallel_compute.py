"""
# DISCLAIMER: This code is provided for academic and educational purposes only.
# Commercial use, redistribution, or any other form of reproduction for profit
# without explicit written consent from the author is strictly prohibited.

# The author of this code is not responsible for any misuse or unauthorized use,
# and assumes no liability for any consequences resulting from the use of this code.

# If you wish to use this code for commercial purposes or have any inquiries
# regarding its use, please contact the author Matthew Glace at glacemk@vcu.edu.

Created by Matthew Glace, 2023

"""

import numpy as np
import concurrent.futures


# Trapezoidal Summation from baseline
def compute_area(data_slice):
    
    # Length of linear baseline
    num_points = len(data_slice)  

    # Create an array with shape (num_points,) ranging from 0 to 1
    interpolation_factor = np.linspace(0, 1, num_points)

    # Interpolate along each row to create the baseline
    baseline = data_slice[0] + (data_slice[-1] - data_slice[0]) * interpolation_factor[:, np.newaxis]

    # Subtract baseline from all the spectra
    corrected_data = data_slice - baseline

    # Compute area solution
    return np.sum(corrected_data, axis=0)



# Batch segmentation operation for each core in feature_matrix2
def process_combination_batch(data, max_range, combinations, batch):
    cfm_batch = np.zeros((len(data[0, 1:]), len(batch)))

    for idx, j in enumerate(batch):
        start, stop = combinations[j]
        data_slice = data[start:stop + 1, 1:]
        
        # Adjust the function to use data_slice and other inputs appropriately
        cfm_batch[:, idx] = compute_area(data_slice)

    return cfm_batch



# Generates feature matrix WITH multicore
def feature_matrix2(data, max_range, batch_size = 20000):     


    # Get number of pixels in IR data (p)
    pixel_count = data.shape[0]
    
    # Generate all combinations of start and stop indices for baseline correction
    ll_start_arr = np.arange(0, pixel_count-2)
    combinations = [(A, B) for A in ll_start_arr for B in range(A+2,pixel_count)]

    # Filter combinations based on range
    valid_combinations = []
    for i, (start, stop) in enumerate(combinations):
        if stop - start <= max_range:
            valid_combinations.append(i)
            

    # Divide valid_combinations into batches
    batches = [valid_combinations[i:i + batch_size] for i in range(0, len(valid_combinations), batch_size)]

    
    # Initialize the process pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=30) as executor:
        futures = []

        # Submit tasks for each batch
        for batch in batches:
            future = executor.submit(process_combination_batch, data, max_range, combinations, batch)
            futures.append(future)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

        # Concatenate the results in the correct order to form cfm
        cfm = np.concatenate([future.result() for future in futures], axis=1)
    
    # Store start and stop indices, and range for each combination
    fm_label = np.zeros((3, len(valid_combinations)))
    fm_label[0] = [combinations[i][0] for i in valid_combinations]
    fm_label[1] = [combinations[i][1] for i in valid_combinations]

    
    return cfm, fm_label
    


# Generates feature matrix WITH multicore
def feature_matrix2b(cal_data, et_data, max_range, batch_size = 20000):     



    data = np.column_stack((cal_data, et_data[:, 1:]))
    
    # Get number of pixels in IR data
    pixel_count = data.shape[0]
    
    # Generate all combinations of start and stop indices for baseline correction
    ll_start_arr = np.arange(0, pixel_count-2)
    combinations = [(A, B) for A in ll_start_arr for B in range(A+2,pixel_count)]

    # Filter combinations based on range
    valid_combinations = []
    for i, (start, stop) in enumerate(combinations):
        if stop - start <= max_range:
            valid_combinations.append(i)
            

    # Divide valid_combinations into batches

    batches = [valid_combinations[i:i + batch_size] for i in range(0, len(valid_combinations), batch_size)]

    
    # Initialize the process pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        futures = []

        # Submit tasks for each batch
        for batch in batches:
            future = executor.submit(process_combination_batch, data, max_range, combinations, batch)
            futures.append(future)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

        # Concatenate the results in the correct order to form cfm
        cfm = np.concatenate([future.result() for future in futures], axis=1)
    
    
    # Store start and stop indices, and range for each combination
    fm_label = np.zeros((3, len(valid_combinations)))
    fm_label[0] = [combinations[i][0] for i in valid_combinations]
    fm_label[1] = [combinations[i][1] for i in valid_combinations]

    
    
    cal_fm = cfm[:len(cal_data[0, 1:]), :]
    et_fm = cfm[len(cal_data[0, 1:]):, :]

    
    return cal_fm, et_fm, fm_label




