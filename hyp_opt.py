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


from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from joblib import parallel_backend






# Tunes hyperparameters for XGB by minimizing RMSE of 5 fold CV using random conditions
def hyp_xgb_randomize(topfeat_cal, y_cal, num_iterations, n_jobs = -1):
    
    """
    
    selects the best XGB hyperparameters using randomization 
    
    Args:
        num_iterations (int, required): number of iterations for hyperparameter optimization
        n_jobs (int, optional) = computational cores to use, defaults to -1 (all)
    
    """
        
    import warnings
    warnings.filterwarnings('ignore', message='No visible GPU is found')
  
    if num_iterations is None:
        raise ValueError("num_iterations must be provided")

    param_grid = {
             'n_estimators': randint(5, 200),
             'learning_rate': [0.5, 0.4, 0.3, 0.2, 0.1, 0.01],
             'max_depth': randint(1, 8),
             'min_child_weight': [1, 5, 10],
             'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
             'gamma': [0, 0.1, 0.2, 0.3, 0.4],
             'reg_alpha': [0, 0.01, 0.1, 1, 10],
             'reg_lambda': [0, 0.01, 0.1, 1, 10],
             'scale_pos_weight': [1, 5, 10],
             'max_delta_step': randint(1, 10),
             'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
             'colsample_bynode': [0.6, 0.7, 0.8, 0.9, 1.0],
             'max_leaves': randint(1, 50),
             }

    xgb_regressor = xgb.XGBRegressor(random_state=42)

    with parallel_backend('loky', n_jobs= n_jobs):
        randomized_search = RandomizedSearchCV(
            estimator=xgb_regressor,
            param_distributions=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_iter= num_iterations,
            verbose=1,
            random_state=42
        )
        randomized_search.fit(topfeat_cal, y_cal)
        
    best_params_xgb = randomized_search.best_params_

    return best_params_xgb



