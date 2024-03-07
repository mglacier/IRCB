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


import sys
sys.path.append('C:\\Users\\spect\\Desktop\\PatternWave\\IRCB - Public\\IRCB')     ################## Change this line for your system



import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
from stats import case_stats



def ml_elr(topfeat_cal, topfeat_test, y_cal, y_test):
        
        
        """
        runs a predictive model of the data using ensemble linear regression
            
        """


        x_cal = topfeat_cal
        x_test = topfeat_test        
        y_cal = y_cal.reshape(-1, 1)
        y_test = y_test
        
        rmse = []
        
        model = LinearRegression()
        y_cal_pred_temp = cross_val_predict(model, x_cal[:,0].reshape(-1,1), y_cal, cv=5)
        y_cal_pred = y_cal_pred_temp
        y_cal_pred_av = y_cal_pred_temp
        rmse.append(root_mean_squared_error(y_cal_pred_av, y_cal))


        for i in range (1,len(x_cal[0,:])):
            y_cal_pred_temp = cross_val_predict(model, x_cal[:,i].reshape(-1,1), y_cal, cv=5)
            y_cal_pred = np.column_stack((y_cal_pred, y_cal_pred_temp))
            y_cal_pred_av = np.mean(y_cal_pred, axis=1)
            rmse.append(root_mean_squared_error(y_cal_pred_av, y_cal))

        lowest_index = rmse.index(min(rmse)) + 1
        
        
        prediction_cal_temp = np.zeros((len(y_cal),lowest_index))
        prediction_test_temp = np.zeros((len(x_test[:,0]),lowest_index))
        prediction_cv_temp = np.zeros((len(y_cal),lowest_index))
        
        
        for i in range (0, lowest_index):
            model = LinearRegression().fit(x_cal[:,i].reshape(-1,1),y_cal)
            prediction_cal_temp[:,i] = model.predict(x_cal[:,i].reshape(-1,1)).flatten()
            prediction_test_temp[:,i] = model.predict(x_test[:,i].reshape(-1,1)).flatten()
            prediction_cv_temp[:,i] = cross_val_predict(model, x_cal[:,i].reshape(-1,1), y_cal, cv=5).flatten()
            
        
        # Make predictions for the calibration and test sets
        prediction_cal = np.mean(prediction_cal_temp, axis=1)
        prediction_test = np.mean(prediction_test_temp, axis=1)
        prediction_cv = np.mean(prediction_cv_temp, axis=1)
 
        stats = case_stats(y_cal, y_test, prediction_cal, prediction_cv, prediction_test)
 
        # Create a dictionary to store the results
        new_result = {
            'model_type' : "ELR",
            'statistics': stats,
            'predictions_cal': prediction_cal.reshape(-1,1),
            'predictions_cv' : prediction_cv.reshape(-1,1),
            'predictions_test': prediction_test.reshape(-1,1),
            'features_included': lowest_index,
            'y_cal' : y_cal.reshape(-1,1),
            'y_test': y_test.reshape(-1,1) if y_test is not None else None,
        }
        
        return new_result 



def ml_xgb(topfeat_cal, topfeat_test, y_cal, y_test, hyperparams):
    
    """
    runs a predictive model of the data using XGB
    
    Args:
        
    """
     

    # Create the best XGBoost Regressor with the best parameters
    best_xgb_regressor = xgb.XGBRegressor(random_state=42, n_jobs = -1, **hyperparams)


    # Fit the best XGBoost model to the calibration set
    best_xgb_regressor.fit(topfeat_cal, y_cal)

    # Make predictions for the calibration and test sets
    prediction_cal = best_xgb_regressor.predict(topfeat_cal).reshape(-1,1)
    prediction_test = best_xgb_regressor.predict(topfeat_test).reshape(-1,1)
    prediction_cv = cross_val_predict(best_xgb_regressor, topfeat_cal, y_cal, cv=5)
    


    stats = case_stats(y_cal, y_test, prediction_cal, prediction_cv, prediction_test)


    # Create a dictionary to store the results
    new_result = {
        'model_type' : "XGB",
        'statistics': stats,
        'predictions_cal': prediction_cal.reshape(-1,1),
        'predictions_cv' : prediction_cv.reshape(-1,1),
        'predictions_test': prediction_test.reshape(-1,1),
        'hyperparameters': hyperparams,
        'y_cal' : y_cal.reshape(-1,1),
        'y_test': y_test.reshape(-1,1) if y_test is not None else None,
    }
    
    return new_result



def ml_rf(topfeat_cal, topfeat_test, y_cal, y_test, hyperparams):
     
     """
     runs a predictive model of the data using sklearn random forest
     
     """
         
     # Create a Random Forest Regressor instance with the best parameters
     best_rf_regressor = RandomForestRegressor(**hyperparams, random_state=42, n_jobs = -1)

     # Fit the model with the best parameters to the calibration set
     best_rf_regressor.fit(topfeat_cal, y_cal.ravel())

     # Make predictions for the calibration and test sets
     prediction_cal = best_rf_regressor.predict(topfeat_cal)
     prediction_test = best_rf_regressor.predict(topfeat_test)
     prediction_cv = cross_val_predict(best_rf_regressor, topfeat_cal, y_cal.ravel(), cv=5)
     
     stats = case_stats(y_cal, y_test, prediction_cal, prediction_cv, prediction_test)

     # Create a dictionary to store the results
     new_result = {
         'model_type' : "RF",
         'statistics': stats,
         'predictions_cal': prediction_cal.reshape(-1,1),
         'predictions_cv' : prediction_cv.reshape(-1,1),
         'predictions_test': prediction_test.reshape(-1,1),
         'hyperparameters': hyperparams,
         'y_cal' : y_cal.reshape(-1,1),
         'y_test': y_test.reshape(-1,1) if y_test is not None else None,
     }
     
     return new_result

