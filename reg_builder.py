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

from parallel_compute import feature_matrix2, feature_matrix2b
from ml_models import ml_elr, ml_rf, ml_xgb
from stats import fold_results_summary
from sorting import regression_fit
from hyp_opt import hyp_xgb_randomize, hyp_rf_gridsearch



# Standard library imports
import math
import os

import warnings
import time


from matplotlib.figure import Figure
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression


with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)



def load_fp(file_path):
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, header=None)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
        else:
            raise ValueError("Unsupported file type. Please use .xlsx or .csv files.")
        return df.to_numpy()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    
    


# Class of functions for sorting and fitting the feature matrix elements
class regression_model_builder:
    
    
    """
    A class for building and tuning predictive models.
 
    Args:
        cal_data (array-like): Training data, with column 0 as label.
        et_data (array-like): Testing data, with column 0 as label.
        comb_data (array-like): Combined test and train data, with column 0 as label
        y_cal (array-like): Training target values.
        y_test (array-like): Testing target values.
        y_comb (array-like): Combined test and train data values

    """
    
    
    def __init__(self, **kwargs):

        

        # Type 2 Model: Requires Later Spliting
        if 'x_comb' in kwargs:
            
            self.x_comb = kwargs.get('x_comb', None)
            self.y_comb = kwargs.get('y_comb', None)
            self.comb_fm = kwargs.get('comb_fm', None)
            
            self.num_components = len(self.y_comb[0,:])
            self.model_type = 2
            
        
        # Type 1 Model: Training and Test are pre-split
        else:
        
            self.x_cal = kwargs.get('x_cal', None)
            self.y_cal = kwargs.get('y_cal', None)
            self.cal_fm = kwargs.get('cal_fm', None)
        
            self.x_test = kwargs.get('x_test', None)
            self.y_test = kwargs.get('y_test', None)
            self.test_fm = kwargs.get('test_fm', None)
        
            self.num_components = len(self.y_cal[0,:])
            self.model_type = 1
            self.n_splits = 1
        
        
        
        self.fm_label = kwargs.get('fm_label', None)
        self.best_params_xgb = None
        self.best_params_rf = None
        self.max_range = 10**6
        self.fit = None
        self.best_indices = None
        self.results = []
        self.criteria = None
        self.percentage = None
        
        
        # self.n_jobs = kwargs.get('n_jobs', -1)
        
        
        self.default_params_rf = {
            'n_estimators': 100,       
            'max_depth': 15,           
            'min_samples_split': 3,    
            'min_samples_leaf': 1,     
            }
        
        
        self.default_params_xgb = {
            'n_estimators': 100, 
            'max_depth': 3
            }
        
        
        
     

    # =========================== X Block Expansion first required =========================== #
    
    
    # Generates the feature matrix for either the calibraton matrix only or both the calibration and test matrices
    def generate_xblock(self):
        
        """
        Applies iterative baseline correction to x-block of calibration and test (optional)

        Returns:
            
        """
        
        if self.model_type == 1:
            self.cal_fm, self.test_fm, self.fm_label = feature_matrix2b(self.x_cal, self.x_test, self.max_range)
        else:
            self.comb_fm, self.fm_label = feature_matrix2(self.x_comb, self.max_range)
          
        return self
        
    
    
    # =========================== Splitting into cal and test when test is not designated =========================== #
    
    

        
    
    
    
    
    
    def split_data(self, test_size = 0.2, random_state = 42):
        
        """
        splits the comb_fm and y_comb into training and testing
        
        Args:
            test_size (float, optional): portion of data for testing. defaults to 0.2
            random_state (int, optional): random seed for splitting, defaults to 42
            
        """
        
        
        
        if self.model_type == 1:
            raise ValueError("Test and training are predeclared, did you mean to split?")
        
        
        self.model_type = 2.1
        self.n_splits = 1
        
        # Assuming X and y are your feature matrix and target vector
        self.cal_fm, self.test_fm, self.y_cal, self.y_test = train_test_split(self.comb_fm, self.y_comb, test_size=test_size, random_state = random_state)
        
        
        # Split your original data for visualization
        transposed_data = np.transpose(self.x_comb[:,1:])
        x_cal_t, x_test_t = train_test_split(transposed_data, test_size=test_size, random_state=random_state)
        self.x_cal = np.hstack((self.x_comb[:, 0].reshape(-1, 1), np.transpose(x_cal_t)))
        self.x_test = np.hstack((self.x_comb[:,0].reshape(-1,1), np.transpose(x_test_t)))

        return self
    
    
    
    def split_kfold(self, n_splits):
        
        
        if self.model_type == 1:
            raise ValueError("Test and training are predeclared, did you mean to split?")
        
        
        self.model_type = 2.2
        self.n_splits = n_splits
        
        # Create KFold object
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # Lists to hold the split datasets
        self.k_cal_fm = []
        self.k_test_fm = []
        self.k_y_cal = []
        self.k_y_test = []
        self.k_x_cal = []
        self.k_x_test = []
        
        
        temp_x_mat = self.x_comb[:,1:].T


        # Iterate over each split
        for train_index, test_index in kf.split(temp_x_mat):


            self.k_x_cal.append(temp_x_mat[train_index])
            self.k_x_test.append(temp_x_mat[test_index])
            
            self.k_y_cal.append(self.y_comb[train_index])
            self.k_y_test.append(self.y_comb[test_index])
                
            if self.comb_fm is not None:
                
                self.k_cal_fm.append(self.comb_fm[train_index])
                self.k_test_fm.append(self.comb_fm[test_index])
                
            
        return self
        
    def no_ircb(self, n_splits):
        
        """
        Split the original combined data into n_splits folds for cross-validation,
        appending the training and testing data for each fold directly to topfeat_cal and topfeat_test,
        with each entry being a list of the same array repeated self.num_components times.
    
        Args:
            n_splits (int): The number of splits for cross-validation.
        """
    
        if self.model_type != 2.2:
            raise ValueError("no_ircb method is only applicable for type 2.2 model (combined data).")
    
    
    
        # Create KFold object
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
        # Reset or initialize the topfeat lists
        self.topfeat_cal = []
        self.topfeat_test = []
    
        # Transpose x_comb excluding the label column to align with comb_fm
        temp_x_mat = self.x_comb[:, 1:].T  # Transpose for correct alignment
    
        # Iterate over each split
        for train_index, test_index in kf.split(temp_x_mat):  # Use .T to operate on original row structure
            # Extract training and testing sets for the current fold
            # Note: After transposing back, use rows for splitting
            x_cal_fold = temp_x_mat[train_index]
            x_test_fold = temp_x_mat[test_index]
    
            # Create lists of the same array repeated self.num_components times for this fold
            cal_fold_repeated = [x_cal_fold for _ in range(self.num_components)]
            test_fold_repeated = [x_test_fold for _ in range(self.num_components)]
    
            # Append these lists to topfeat_cal and topfeat_test
            self.topfeat_cal.append(cal_fold_repeated)
            self.topfeat_test.append(test_fold_repeated)
    
        return self

    

    # =========================== Regression fitting for expanded x-block =========================== #
    
    # Fits 1st order regression to expanded X-block columns, returns RMSE
    def regression_fit(self, criteria = 'RMSE'):
        
        """
        Fits a first order polynomial to each x-block column

        Returns:  
        arr: A tuple containing the RMSE for each column
        
        """
        
        self.criteria = criteria
        self.fit = []
        
        # For a simple test train split
        if self.model_type == 1 or self.model_type == 2.1:
      
            split_fit = []
            for j in range(self.n_splits):
                for i in range(0,self.num_components):
                    split_fit.append(regression_fit(self.cal_fm,self.y_cal[:,i], self.criteria))
                self.fit.append(split_fit)
            
            
        # For k-fold splitting
        elif self.model_type == 2.2:
      
            split_fit = []
            for j in range(self.n_splits):
                for i in range(0,self.num_components):
                    split_fit.append(regression_fit(self.k_cal_fm[j],self.k_y_cal[j][:,i], self.criteria))
                self.fit.append(split_fit)
                
        elif self.model_type == 2:
            raise ValueError("Split the data before regression fitting")
      
        
        return self
    
    


    # =========================== Returns a portion of the expanded x-block based on the regression fit  =========================== #

    # Returns the top x elements of the feature matrix
    def reduce_xblock(self, percentage):
        
        """
        Returns the top portion of feature matrix.

        Args:
        percentage (float): The portion of feature matrix to return.

        Returns:
        arr: A arrary containing the top feature matrices for calibration and test data (optional)

        """
        
        
        self.percentage = percentage

        # For a simple test train split
        if self.model_type == 1 or self.model_type == 2.1:

            # Establishes the % of solution array to be included
            num_solutions = math.floor(len(self.cal_fm[0, :])*percentage)
    
    
            self.best_indices = []
            self.topfeat_cal = []
            self.topfeat_test = []
    
    
    
            best_indices_temp = []
            topfeat_cal_temp = []
            topfeat_test_temp = []
            
            
            for i in range(0,self.num_components):
                
                # Find the indices of the num_solutions combinations with the lowest quality metric
                best_indices = np.argsort(self.fit[0][i][-1])[:num_solutions]
            
                topfeat_cal = np.zeros((len(self.cal_fm[:, 0]), num_solutions))
                topfeat_test = np.zeros((len(self.test_fm[:, 0]), num_solutions))
    
                topfeat_cal = self.cal_fm[:, best_indices]
                topfeat_test = self.test_fm[:, best_indices]
            
                best_indices_temp.append(best_indices)
                topfeat_cal_temp.append(topfeat_cal)
                topfeat_test_temp.append(topfeat_test)
            
            
            self.best_indices.append(best_indices_temp)
            self.topfeat_cal.append(topfeat_cal_temp)
            self.topfeat_test.append(topfeat_test_temp)
        
        
        
        
        
        # For k-fold splitting
        elif self.model_type == 2.2:
            
            # Establishes the % of solution array to be included
            num_solutions = math.floor(len(self.k_cal_fm[0][0, :])*percentage)
            
            
            self.best_indices = []
            self.topfeat_cal = []
            self.topfeat_test = []
            
            
            for j in range(self.n_splits):
                
                best_indices_temp = []
                topfeat_cal_temp = []
                topfeat_test_temp = []
                
                
                for i in range(0,self.num_components):
                    
                    # Find the indices of the num_solutions combinations with the lowest quality metric
                    best_indices = np.argsort(self.fit[j][i][-1])[:num_solutions]
                
                    topfeat_cal = np.zeros((len(self.k_cal_fm[j][:, 0]), num_solutions))
                    topfeat_test = np.zeros((len(self.k_test_fm[j][:, 0]), num_solutions))
        
                    topfeat_cal = self.k_cal_fm[j][:, best_indices]
                    topfeat_test = self.k_test_fm[j][:, best_indices]
                
                    best_indices_temp.append(best_indices)
                    topfeat_cal_temp.append(topfeat_cal)
                    topfeat_test_temp.append(topfeat_test)
                
                
                self.best_indices.append(best_indices_temp)
                self.topfeat_cal.append(topfeat_cal_temp)
                self.topfeat_test.append(topfeat_test_temp)
        
        
        return self




    # =========================== Tunes the Hyperparameters for Regression (optional)  =========================== #
    


    # Tunes hyperparameters for XGB by minimizing RMSE of 5 fold CV using random conditions
    def opt_xgb_randomize(self, num_iterations, param_grid = None, n_jobs = -1):
        
        
        self.best_params_xgb = []
        
    
        # For a simple test train split
        if self.model_type == 1 or self.model_type == 2.1:
            for i in range(0,self.num_components):
                fold_result = []
                fold_result.append(hyp_xgb_randomize(self.topfeat_cal[0][i], self.y_cal[:,i], num_iterations, n_jobs = -1))
            
                self.best_params_xgb.append(fold_result)
                    
            
        # For k-fold splitting
        elif self.model_type == 2.2:
            for i in range(0,self.num_components):
                fold_result = []
                for j in range(self.n_splits):
                    fold_result.append(hyp_xgb_randomize(self.topfeat_cal[j][i], self.k_y_cal[j][:,i], num_iterations, n_jobs = -1))
                self.best_params_xgb.append(fold_result)
            
        return self
    
    
    
    def opt_rf_gridsearch(self):
        
       
        self.best_params_rf = []
        
        # For a simple test train split
        if self.model_type == 1 or self.model_type == 2.1:
            for i in range(0,self.num_components):
                fold_result = []
                fold_result.append(hyp_rf_gridsearch(self.topfeat_cal[0][i], self.y_cal[:,i], n_jobs = -1))
                self.best_params_rf.append(fold_result)
                    
        # For k-fold splitting
        elif self.model_type == 2.2:
            for i in range(0,self.num_components):
                fold_result = []
                for j in range(self.n_splits):
                    fold_result.append(hyp_rf_gridsearch(self.topfeat_cal[j][i], self.k_y_cal[j][:,i], n_jobs = -1))
                self.best_params_rf.append(fold_result)
        
        return self
        
    

    # =========================== Predictive Models  =========================== #, ########## ADD Self.Percentages to results

    
    def predictive_model(self, ml_type):
    
        """
        Runs a predictive model based on the specified machine learning type and class properties.
    
        This function supports various machine learning models like XGBoost (xgb), Random Forest (rf),
        Elastic Net Regularized Linear Regression (elr), and Partial Least Squares Regression (pls).
        It handles different types of data splitting strategies (e.g., simple train-test split, k-fold splitting)
        and uses the best available parameters for each model or defaults if none are provided.
    
        Parameters:
        - ml_type (str): The type of machine learning model to run. Accepted values are 'xgb', 'rf', 'elr'
    
        The function iterates over the number of components (self.num_components) and applies the specified
        machine learning model to each. It aggregates the results for each component, including a summary
        and detailed fold results, and appends them to the class's results attribute.
    
        The function uses various class attributes for its operations:
        - model_type: Defines the model strategy (e.g., simple split, k-fold).
        - best_params_xgb, best_params_rf: Best parameters for XGBoost and Random Forest models.
        - default_params_xgb, default_params_rf: Default parameters for XGBoost and Random Forest models.
        - topfeat_cal, topfeat_test: Features for calibration and testing.
        - y_cal, y_test: Target variables for calibration and testing.
        - k_y_cal, k_y_test: Target variables for k-fold splits.
    
        Returns:
        None. The function updates the class's results attribute with the results of the modeling.
    
        Note:
        This function prints a message if default parameters are used for XGBoost or Random Forest models.
        """
        
        component_results = {}
       
        component_results["model_info"] = {
           "ml_type": ml_type,
           "model_type": self.model_type,
           "max_range": self.max_range,
           "criteria": self.criteria,
           "percentage": self.percentage,
           "best_indices": self.best_indices,
           "best_params_xgb": self.best_params_xgb,
           "default_params_xgb": self.default_params_xgb,
           "best_params_rf": self.best_params_rf,
           "default_params_rf": self.default_params_rf,
           "num_components": self.num_components,
           "topfeat_cal": self.topfeat_cal,
           "topfeat_test": self.topfeat_test
           }
       
       
    
        for i in range(0,self.num_components):
           
           fold_result = {}
           
          
           if self.model_type == 1 or self.model_type == 2.1:                       # For a simple test/train split
               
               if ml_type == 'xgb':
                   
                   if self.best_params_xgb is not None:
                        params = self.best_params_xgb[i][0] 
                   else :
                        params = self.default_params_xgb
                        print('setting default xgb hyperparameters')
                 
                   if self.y_test is not None:
                       fold_result["split1"] = ml_xgb(self.topfeat_cal[0][i], self.topfeat_test[0][i], self.y_cal[:,i], self.y_test[:,i], params)
                   else:
                       fold_result["split1"] = ml_xgb(self.topfeat_cal[0][i], self.topfeat_test[0][i], self.y_cal[:,i], None, params)

               elif ml_type == 'rf':
                   
                   if self.best_params_rf is not None:
                       params = self.best_params_rf[i][0] 
                   else :
                       params = self.default_params_rf
                       print('setting default rf hyperparameters')
                   
                   if self.y_test is not None:
                       fold_result["split1"] = ml_rf(self.topfeat_cal[0][i], self.topfeat_test[0][i], self.y_cal[:,i], self.y_test[:,i], params)
                   else:
                       fold_result["split1"] = ml_rf(self.topfeat_cal[0][i], self.topfeat_test[0][i], self.y_cal[:,i], None, params)
                   
                   
               elif ml_type == 'elr':
        
                   if self.y_test is not None:
                       fold_result["split1"] = ml_elr(self.topfeat_cal[0][i], self.topfeat_test[0][i], self.y_cal[:,i], self.y_test[:,i])
                   else:
                       fold_result["split1"] = ml_elr(self.topfeat_cal[0][i], self.topfeat_test[0][i], self.y_cal[:,i], None)



                   
                   
           # For k-fold splitting
           elif self.model_type == 2.2:
               
               for j in range(self.n_splits):
                   
                   split_name = f"split{j+1}"
                   
                   if ml_type == 'xgb':
                   
                       if self.best_params_xgb is not None:
                            params = self.best_params_xgb[i][j] 
                       else :
                            params = self.default_params_xgb
                            print('setting default xgb hyperparameters')
               
                       fold_result[split_name] = ml_xgb(self.topfeat_cal[j][i], self.topfeat_test[j][i], self.k_y_cal[j][:,i], self.k_y_test[j][:,i], params)
        
                   elif ml_type == 'rf':
                    
                       if self.best_params_rf is not None:
                            params = self.best_params_rf[i][j] 
                       else :
                            params = self.default_params_rf
                            print('setting default rf hyperparameters')
                       
                       fold_result[split_name] = ml_rf(self.topfeat_cal[j][i], self.topfeat_test[j][i], self.k_y_cal[j][:,i], self.k_y_test[j][:,i], params)
                       
                       
                   elif ml_type == 'elr':
                       
                       fold_result[split_name] = ml_elr(self.topfeat_cal[j][i], self.topfeat_test[j][i], self.k_y_cal[j][:,i], self.k_y_test[j][:,i])
                       
     
    
           temp_result = {
               "summary": fold_results_summary(fold_result),
               "fold_results": fold_result,
               }
            
           key_name = f"Compound{i+1}"
           component_results[key_name] = temp_result
           
           
        self.results.append(component_results)
       
       
       

# Class of functions for visualizing the modelling results
class visualizer:
    
    """
    a class for visualizing the modelling results
    
    Args:
        model (object): completed model to analyze
        
    """
    
    def __init__(self, model, analyte = 0, fold = 0):

 
        self.model_type = model.model_type
        self.analyte = analyte
        self.fold = fold
        self.results = model.results
        self.fm_label = model.fm_label
        self.fit = model.fit
        self.topfeat_cal = model.topfeat_cal
        self.topfeat_test = model.topfeat_test

        
        
        if self.model_type == 1:
        
            self.x_cal = model.x_cal
            self.y_cal = model.y_cal[:,self.analyte]
        
            self.x_test = model.x_test
            
            if model.y_test is not None:
                self.y_test = model.y_test[:, self.analyte]
            else:
                self.y_test = None

        
        
        
        elif self.model_type == 2.1:
            
            self.x_comb = model.x_comb
            self.y_comb = model.y_comb[:,self.analyte]
            
            self.cal_data = model.x_cal
            self.y_cal = model.y_cal[:,self.analyte]
        
            self.et_data = model.x_test
            
            if model.y_test is not None:
                self.y_test = model.y_test[:, self.analyte]
            else:
                self.y_test = None

            
        elif self.model_type == 2.2:

            
            self.x_comb = model.x_comb
            self.y_comb = model.y_comb[:, self.analyte]
            
    
            label = self.x_comb[:, 0]
            data_wo_label = self.x_comb[:, 1:].T
            

            kf = KFold(n_splits=model.n_splits, shuffle=True, random_state=42)
            
            # Lists to hold the split datasets
            k_cal_data = []
            k_et_data = []
            k_y_cal = []
            k_y_test = []
            
            # Iterate over each split
            for train_index, test_index in kf.split(data_wo_label):

                # Append data to lists, transposing if necessary
                k_cal_data.append(data_wo_label[train_index].T)
                k_et_data.append(data_wo_label[test_index].T)

                k_y_cal.append(self.y_comb[train_index])
                k_y_test.append(self.y_comb[test_index])
                
                
            self.x_cal = np.hstack((label.reshape(-1,1),k_cal_data[self.fold]))
            self.y_cal = k_y_cal[self.fold]
        
            self.x_test = np.hstack((label.reshape(-1,1),k_et_data[self.fold]))
            self.y_test = k_y_test[self.fold]
            
            


          
        
        # Sorts the fm_label according to fit
        def sort_fm_label(fm_label, fit, fold = self.fold, analyte = self.analyte):
        
           fm_stacked = np.vstack((fm_label,fit[fold][analyte]))
        
           # Getting the order of indices that would sort the last row
           sorted_indices = np.argsort(fm_stacked[-1, :])

           # Sorting the entire matrix according to these indices
           fm_label_sorted = fm_stacked[:, sorted_indices]
            
    
           return fm_label_sorted
        
        self.fm_label_sorted = sort_fm_label(self.fm_label, self.fit, self.fold, self.analyte)
    
    

        
        
    def plot_baselines(self, save=False, x=1, x_limits=None, y_limits=None):
        """
        Plots the best x number of baselines
    
        Args:
            x (int): number of baselines to plot, defaults to 1
            x_limits: plot limits for x-axis, defaults to None
            y_limits: plot limits for y-axis, defaults to None
        """
    
        start = self.fm_label_sorted[0, 0:x]
        stop = self.fm_label_sorted[1, 0:x]
    
        data = self.x_cal
    
        start = np.round(np.array(start, dtype='float'))
        stop = np.round(np.array(stop, dtype='float'))
        start_array = start.astype('int')
        stop_array = stop.astype('int')
    
        baselines = []
    
        for start, stop in zip(start_array, stop_array):
            baseline = ((data[stop, 1:] - data[start, 1:]) / (stop - start)) * (np.arange(start, stop + 1) - start)[:, np.newaxis] + data[start, 1:]
            baselines.append(baseline)
    
        temp_data_list = []
    
        for start, stop in zip(start_array, stop_array):
            temp_data = data[start:stop + 1, :]
            temp_data_list.append(temp_data)
    
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
        fig, ax = plt.subplots(figsize=(8, 6))
    
        for i, baseline in enumerate(baselines):
            x_baseline = temp_data_list[i][:, 0]
            y_baseline = baseline[:, :]
            ax.plot(x_baseline, y_baseline, '--', linewidth=0.5, color=colors[i % len(colors)])
    
        x_data = data[:, 0]
        y_data = data[:, 1:]
    
        for i, y in enumerate(y_data.T):
            ax.plot(x_data, y, color='grey', linewidth=0.2)
    
        ax.set(xlabel='Raman Shift (cm$^{-1}$)', ylabel='Intensity (A.U.)')
    
        if x_limits is not None:
            ax.set_xlim(x_limits[0], x_limits[1])
    
        if y_limits is not None:
            ax.set_ylim(y_limits[0], y_limits[1])
    
        ax.invert_xaxis()
    
        if save:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"Baselines_{timestamp}.png"
            desktop_path = os.path.expanduser("~/Desktop")
            filepath = os.path.join(desktop_path, filename)
            fig.savefig(filepath, dpi=800, bbox_inches="tight", format="png")
    
        return fig

    

    def regression_plot(self, save=False, analyte_name=None, number=0):
        """
        Plots the result of the model.
    
        Args:
            analyte_name (str, optional): Name to put on plot title.
            number (int, optional): Plot model # from results, defaults to 0.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
        
            # Create a Figure object
            fig = Figure(figsize=(5, 4))
            ax = fig.add_subplot(111)  # Add a subplot to the figure
        
            # Assuming self.analyte is a numeric value
            compound = f'Compound{self.analyte + 1}'
        
            predictions_cal = self.results[number][compound]['summary']['predictions_cal'].flatten()
            y_cal = self.results[number][compound]['summary']['y_cal'].flatten()
            predictions_test = self.results[number][compound]['summary']['predictions_test'].flatten()
            y_test = self.results[number][compound]['summary']['y_test'].flatten() if self.results[number][compound]['summary']['y_test'] is not None else None
        
            model = sm.OLS(y_cal, sm.add_constant(predictions_cal)).fit()
            cal_eqn = f'Calibration Set: y = {model.params[1]:.3f}x + {model.params[0]:.3f}'
        
            if self.model_type == 2.2 and y_test is not None:
                sns.regplot(x=predictions_test, y=y_test, ax=ax, label='Test Set', ci=None, marker='x', color='orange', line_kws={'linewidth': 1, 'linestyle': '--', 'color': 'blue'})
                test_eqn = f'K-fold Test Set: y = {model.params[1]:.3f}x + {model.params[0]:.3f}'
                ax.text(0.2, 0.02, test_eqn, va='bottom', transform=ax.transAxes, fontsize=12)
            else:
                sns.regplot(x=predictions_cal, y=y_cal, ax=ax, label='Calibration Set', ci=None, line_kws={'linewidth': 1, 'linestyle': '--'})
                if y_test is not None:
                    sns.scatterplot(x=predictions_test, y=y_test, ax=ax, label='Test Set', marker='x', s=100, color='orange')
                    test_model = sm.OLS(y_test, sm.add_constant(predictions_test)).fit()
                    test_eqn = f'Test Set: y = {test_model.params[1]:.3f}x + {test_model.params[0]:.3f}'
                    ax.text(0.35, 0.025, cal_eqn + '\n' + test_eqn, va='bottom', transform=ax.transAxes, fontsize=12)
                else:
                    ax.text(0.35, 0.025, cal_eqn, va='bottom', transform=ax.transAxes, fontsize=12)
        
            if analyte_name is not None:
                ax.set_title(f"{analyte_name} Regression Model")
        
            ax.set_xlabel('Prediction Values')
            ax.set_ylabel('Reference Values')
            max_x = max(max(predictions_cal), max(predictions_test) if predictions_test.size > 0 else 0)
            max_y = max(max(y_cal), max(y_test) if y_test is not None and y_test.size > 0 else 0)
            ax.set_xlim(0, max_x * 1.2)
            ax.set_ylim(0, max_y * 1.2)
            ax.legend(frameon=True, loc='upper left', edgecolor='black')
        
            if save:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"RegressionPlot_{timestamp}.png"
                desktop_path = os.path.expanduser("~/Desktop")
                filepath = os.path.join(desktop_path, filename)
                fig.savefig(filepath, dpi=800, bbox_inches="tight", format="png")
        
        return fig



    # Returns the regression statistics for the solution array (Baselines)
    def baseline_report(self): 

        """
        generates a report about the fit of each baseline
        
        """        


        calibration_info = []
        num_baselines = len(self.topfeat_cal[self.fold][self.analyte][0, :])
        
   
        axis_label = self.x_cal[:,0].flatten()
        

            
        
        for i in range(num_baselines):
            
            # Fit the linear regression model
            model = LinearRegression()
            x = self.topfeat_cal[self.fold][self.analyte][:, i].reshape(-1,1)
            y = self.y_cal.reshape(-1,1)
            model.fit(x, y)

            # Predict the concentrations using the fitted polynomial
            y_pred = model.predict(x)
            
            rmse = root_mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            lower = self.fm_label_sorted[0, i].astype(int)
            upper = self.fm_label_sorted[1, i].astype(int)

            # Append calibration information to the list
            calibration_info.append([
                 axis_label[lower].round(1),
                 axis_label[upper].round(1),
                 r2.round(3),
                 rmse.round(3),
                 model.coef_[0][0].round(4),
                 model.intercept_[0].round(4)
                 ])
           
        # Create header for the calibration report
        header = np.array(['Limit 1', 'Limit 2', 'R^2', 'RMSE', 'Coefficient(s)', 'Intercept'])

        # Create the calibration report as a NumPy array
        self.baseline_stats = np.vstack((header, np.array(calibration_info, dtype=object)))
        
        return self.baseline_stats
        






    
    
    



