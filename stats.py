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
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)


def case_stats(y_cal, y_test, prediction_cal, prediction_cv, prediction_test):
    
    def pearson_r2_score(y_true, y_pred):
        r, _ = pearsonr(y_true.ravel(), y_pred.ravel())
        return r ** 2  # Square the Pearson correlation coefficient
    
    # Calculate metrics for calibration and cross-validation sets
    metrics = {
        'RMSE': {
            'cal': root_mean_squared_error(y_cal, prediction_cal),
            'cv': root_mean_squared_error(y_cal, prediction_cv)
         },
         
        'R2': {
            'cal': pearson_r2_score(y_cal, prediction_cal),
            'cv': pearson_r2_score(y_cal, prediction_cv)
        },
         
        'MAE': {
            'cal': mean_absolute_error(y_cal, prediction_cal),
            'cv': mean_absolute_error(y_cal, prediction_cv)
        },
        'Bias': {
            'cal': np.mean(prediction_cal - y_cal),
            'cv': np.mean(prediction_cv - y_cal)
        }
    }

    # Check if y_test is provided and calculate test metrics
    if y_test is not None:
        metrics['RMSE']['test'] = root_mean_squared_error(y_test, prediction_test)
        metrics['R2']['test'] = pearson_r2_score(y_test, prediction_test)
        metrics['MAE']['test'] = mean_absolute_error(y_test, prediction_test)
        metrics['Bias']['test'] = np.mean(prediction_test - y_test)

    # Prepare the result matrix
    labels = list(metrics.keys())
    cal_values = [metrics[label]['cal'] for label in labels]
    cv_values = [metrics[label]['cv'] for label in labels]
    test_values = [metrics[label].get('test', 'N/A') for label in labels]  # Use 'N/A' if test data is not available

    # Stack the data into a 2D array
    stats = np.column_stack((labels, cal_values, cv_values, test_values))

    # Optionally, you can add column labels as follows:
    column_labels = np.array(['Metric', 'Calibration', 'Cross-Validation', 'Test'])
    stats = np.vstack((column_labels, stats))

    return stats



def fold_results_summary(fold_result):
    
    
    n_folds = len(fold_result.items())
    
    y_test = []
    y_cal = []
    
    
    prediction_cal_list = []
    prediction_cv_list = []
    prediction_test_list = []
    
    
    # Looping all of the fold results
    for key, inner_dict in fold_result.items():

        y_cal.append(inner_dict['y_cal'].flatten())
        
        if inner_dict['y_test'] is not None:
            y_test.append(inner_dict['y_test'].flatten())
        else:
            y_test.append(None)  


        prediction_cal_list.append(inner_dict['predictions_cal'].flatten())
        prediction_cv_list.append(inner_dict['predictions_cv'].flatten())
        prediction_test_list.append(inner_dict['predictions_test'].flatten())
    
    
    
    y_cal = np.hstack(y_cal).reshape(-1,1)

    if any(element is None for element in y_test):
        y_test = None
    else:
        y_test = np.hstack(y_test).reshape(-1, 1)
        
    prediction_cal = np.hstack(prediction_cal_list).reshape(-1,1)
    prediction_cv = np.hstack(prediction_cv_list).reshape(-1,1)
    prediction_test = np.hstack(prediction_test_list).reshape(-1,1)
    
    stats = case_stats(y_cal, y_test, prediction_cal, prediction_cv, prediction_test)

    # Create a dictionary to store the results
    summary = {
        'statistics': stats,
        'n_folds': n_folds,
        'predictions_cal': prediction_cal.reshape(-1,1),
        'predictions_cv' : prediction_cv.reshape(-1,1),
        'predictions_test': prediction_test.reshape(-1,1),
        'y_cal' : y_cal.reshape(-1,1),
        'y_test': y_test.reshape(-1,1) if y_test is not None else None,
    }
    
    return summary



