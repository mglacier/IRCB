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
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score





# Regresses each feature matrix factor against the offline determined concentrations
def regression_fit(cal_fm, y_cal, criteria = 'R2', n_jobs = -1):
    
    fit = np.zeros((1, len(cal_fm[1, :])))
    y_cal = y_cal.reshape(-1, 1)

    if criteria == 'R2':
        
        def calculate_fit(i):
            x = cal_fm[:, i].reshape(-1,1)
            model = LinearRegression().fit(x, y_cal)
            model_pred = model.predict(x)
            r2 = r2_score(y_cal, model_pred)
            return 1 - r2

    
    iterations = len(cal_fm[0, :])
    fit[0, :] = Parallel(n_jobs=n_jobs)(delayed(calculate_fit)(i) for i in range(iterations))

    return fit


