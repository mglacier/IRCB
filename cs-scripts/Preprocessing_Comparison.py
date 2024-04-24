#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:50:45 2024

@author: glacier
"""



import sys


sys.path.append('C:\\Users\\spect\\Desktop\\PatternWave\\IRCB - Public\\IRCB')     ################## Change this line for your system


from reg_builder import load_fp
from reg_builder import regression_model_builder, visualizer
from ml_models import ml_elr, ml_rf, ml_xgb



#### Case Study 1

import numpy as np
from scipy.signal import savgol_filter



def svg(matrix, window_length, polyorder):
    """
    Apply a Savitzky-Golay filter to each column in a matrix.
    
    Parameters:
    - matrix: A 2D numpy array where each column represents a spectrum.
    - window_length: The length of the filter window (i.e., the number of coefficients). 
                      window_length must be a positive odd integer.
    - polyorder: The order of the polynomial used to fit the samples. 
                 polyorder must be less than window_length.

    Returns:
    - A 2D numpy array of the same shape as matrix, with the Savitzky-Golay filter applied to each column.
    """
    filtered_matrix = np.apply_along_axis(savgol_filter, 0, matrix, window_length, polyorder)
    return filtered_matrix



# Apply the Savitzky-Golay filter
window_length = 5  # Must be a positive odd integer
polyorder = 2  # The polynomial order





def numerical_second_derivative(matrix):
    """
    Compute the second derivative of each spectrum in a matrix numerically.

    Parameters:
    - matrix: A 2D numpy array where each column represents a spectrum.

    Returns:
    - A 2D numpy array of the same shape as matrix, with the second derivative applied to each column.
    """
    # Allocate space for the second derivative matrix
    second_derivative_matrix = np.zeros_like(matrix)

    # Compute the second derivative for each spectrum
    for i in range(1, matrix.shape[0] - 1):
        second_derivative_matrix[i, :] = matrix[i-1, :] - 2*matrix[i, :] + matrix[i+1, :]
    
    # Handle the boundary cases
    second_derivative_matrix[0, :] = second_derivative_matrix[1, :]
    second_derivative_matrix[-1, :] = second_derivative_matrix[-2, :]

    return second_derivative_matrix





def msc_correction(spectra):
    """
    Perform Multiplicative Scatter Correction (MSC) on spectral data.
    
    Parameters:
    - spectra: A 2D numpy array where rows are samples and columns are spectral data points.

    Returns:
    - msc_spectra: A 2D numpy array with MSC applied.
    """
    # Calculate the mean spectrum across samples (rows)
    mean_spectrum = np.mean(spectra, axis=0)
    
    # Initialize an array to hold the corrected spectra
    msc_spectra = np.zeros_like(spectra)
    
    # Apply MSC to each spectrum
    for i in range(spectra.shape[0]):
        # Perform linear regression between the mean spectrum and each sample spectrum
        fit = np.polyfit(mean_spectrum, spectra[i, :], 1)
        slope, intercept = fit
        
        # Apply the correction to each spectrum
        msc_spectra[i, :] = (spectra[i, :] - intercept) / slope
    
    return msc_spectra








#### Case Study 2

cs2_results_no_IRCB = []



base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/nit/" # Change for your System


x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


    
cs2_results_no_IRCB.append(ml_rf(x_cal2.T,x_test2.T,y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))



base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/ciso/" # Change for your System


x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


    
cs2_results_no_IRCB.append(ml_rf(x_cal2.T,x_test2.T,y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))


base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/cec/" # Change for your System


x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


    
cs2_results_no_IRCB.append(ml_rf(x_cal2.T,x_test2.T,y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))





cs2_results_SVG = []



base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/nit/" # Change for your System


x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


    
cs2_results_SVG.append(ml_rf(svg(x_cal2.T, window_length, polyorder),svg(x_test2.T, window_length, polyorder),y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))



base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/ciso/" # Change for your System


x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


    
cs2_results_SVG.append(ml_rf(svg(x_cal2.T, window_length, polyorder),svg(x_test2.T, window_length, polyorder),y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))


base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/cec/" # Change for your System



x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


cs2_results_SVG.append(ml_rf(svg(x_cal2.T, window_length, polyorder),svg(x_test2.T, window_length, polyorder),y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))






cs2_results_2nd_der = []



base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/nit/" # Change for your System


x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


    
cs2_results_2nd_der.append(ml_rf(numerical_second_derivative(x_cal2.T),numerical_second_derivative(x_test2.T),y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))



base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/ciso/" # Change for your System


x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


    
cs2_results_2nd_der.append(ml_rf(numerical_second_derivative(x_cal2.T),numerical_second_derivative(x_test2.T),y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))


base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/cec/" # Change for your System



x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


cs2_results_2nd_der.append(ml_rf(numerical_second_derivative(x_cal2.T),numerical_second_derivative(x_test2.T),y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))





cs2_results_msc_correction = []



base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/nit/" # Change for your System


x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


    
cs2_results_msc_correction.append(ml_rf(msc_correction(x_cal2.T),msc_correction(x_test2.T),y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))



base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/ciso/" # Change for your System


x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


    
cs2_results_msc_correction.append(ml_rf(msc_correction(x_cal2.T),msc_correction(x_test2.T),y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))


base_path2 = "/Users/glacier/Documents/GitHub/IRCB/cs-data/soil-nir/cec/" # Change for your System



x_cal2 = load_fp(base_path2 + "x_cal.xlsx")[:,1:]
x_test2 = load_fp(base_path2 + "x_test.xlsx")[:,1:]
y_cal2 = load_fp(base_path2 + "y_cal.xlsx")
y_test2 = load_fp(base_path2 + "y_test.xlsx")


cs2_results_msc_correction.append(ml_rf(msc_correction(x_cal2.T),msc_correction(x_test2.T),y_cal2.reshape(-1,1),y_test2.reshape(-1,1), hyperparams = {
        'n_estimators': 100,       
        'max_depth': 15,           
        'min_samples_split': 3,    
        'min_samples_leaf': 1,     
        }))




#### Case Study 3



# IRCB with default hyperparams

base_path = "/Users/glacier/Documents/GitHub/IRCB/cs-data/dense-slurry/" # Change for your System


x_comb = load_fp(base_path + "Raman_x_block.csv")
y_comb = load_fp(base_path + "Raman_y_block.csv")


nuc_solid_model = regression_model_builder(x_comb = x_comb, y_comb = y_comb)


# IRCB


nuc_solid_model.generate_xblock()
nuc_solid_model.split_kfold(n_splits = 10)
nuc_solid_model.regression_fit(criteria='R2')
nuc_solid_model.reduce_xblock(percentage = .02)
nuc_solid_model.predictive_model(ml_type='xgb')
results = nuc_solid_model.results



# No Preprocessing


nuc_solid_model = regression_model_builder(x_comb = x_comb, y_comb = y_comb)
nuc_solid_model.split_kfold(n_splits = 10)
nuc_solid_model.no_ircb(n_splits = 10)
nuc_solid_model.predictive_model(ml_type='xgb')
results_nopre = nuc_solid_model.results




# SVG



# Apply the Savitzky-Golay filter to exclude the wavenumber column
x_comb_svg_filtered = svg(x_comb[:, 1:], window_length=5, polyorder=2)
x_comb_svg = np.hstack((x_comb[:, 0].reshape(-1, 1), x_comb_svg_filtered))



nuc_solid_model = regression_model_builder(x_comb = x_comb_svg, y_comb = y_comb)
nuc_solid_model.split_kfold(n_splits = 10)
nuc_solid_model.no_ircb(n_splits = 10)
nuc_solid_model.predictive_model(ml_type='xgb')
results_svg = nuc_solid_model.results







# Compute the second derivative of the original spectra (excluding wavenumber column)
x_comb_2nd_deriv = numerical_second_derivative(x_comb[:, 1:])
x_comb_2nd = np.hstack((x_comb[:, 0].reshape(-1, 1), x_comb_2nd_deriv))



nuc_solid_model = regression_model_builder(x_comb = x_comb_2nd, y_comb = y_comb)
nuc_solid_model.split_kfold(n_splits = 10)
nuc_solid_model.no_ircb(n_splits = 10)
nuc_solid_model.predictive_model(ml_type='xgb')
results_2nd = nuc_solid_model.results





# Apply MSC to the original spectra (excluding wavenumber column)
x_comb_msc_corrected = msc_correction(x_comb[:, 1:])
x_comb_msc = np.hstack((x_comb[:, 0].reshape(-1, 1), x_comb_msc_corrected))



nuc_solid_model = regression_model_builder(x_comb = x_comb_msc, y_comb = y_comb)
nuc_solid_model.split_kfold(n_splits = 10)
nuc_solid_model.no_ircb(n_splits = 10)
nuc_solid_model.predictive_model(ml_type='xgb')
results_msc = nuc_solid_model.results




