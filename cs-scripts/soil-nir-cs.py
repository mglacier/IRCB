"""
# DISCLAIMER: This code is provided for academic and educational purposes only.
# Commercial use, redistribution, or any other form of reproduction for profit
# without explicit written consent from the author is strictly prohibited.

# The author of this code is not responsible for any misuse or unauthorized use,
# and assumes no liability for any consequences resulting from the use of this code.

# If you wish to use this code for commercial purposes or have any inquiries
# regarding its use, please contact the author Matthew Glace at glacemk@vcu.edu.

Created by Matthew Glace, 2023



INSTRUCTIONS:
    
    TO UTILIZE MULTICORE PROCESSING: HIGHLIGHT LINES AND RUN SELECTION
    
    WARNING: PRESSING PLAY WILL DEFAULT TO ONE CORE

"""


"""
This case study uses RF to generate regression predictions for soil samples from the chimiometrie 2006 challenge
The test set is used according to conference outlines
"""


import sys
sys.path.append('C:\\Users\\spect\\Desktop\\PatternWave\\IRCB - Public\\IRCB')   ################## Change this line for your system
from reg_builder import load_fp
from reg_builder import regression_model_builder, visualizer



# Loads data from Chemometrics and Intelligent Laboratory Systems Volume 91, Issue 1, 15 March 2008, Pages 94-98
base_path = 'C:/Users/spect/Desktop/PatternWave/IRCB - Public/IRCB/cs-data/soil-nir/'  ################## Change this line for your system





# Nitrogen

x_cal = load_fp(base_path + "nit/x_cal.xlsx")
x_test = load_fp(base_path + "nit/x_test.xlsx")
y_cal = load_fp(base_path + "nit/y_cal.xlsx")
y_test = load_fp(base_path + "nit/y_test.xlsx")


nitrogen_model = regression_model_builder(x_cal = x_cal, x_test = x_test, y_cal = y_cal, y_test = y_test)
nitrogen_model.generate_xblock()
nitrogen_model.regression_fit(criteria = 'R2')
nitrogen_model.reduce_xblock(percentage = 0.02)
nitrogen_model.predictive_model(ml_type='rf')

nit_results = nitrogen_model.results

nitrogen_model_analysis = visualizer(nitrogen_model)
nitrogen_model_analysis.plot_baselines(x = 1, save = False)
nitrogen_model_analysis.regression_plot(save = False)
nitrogen_baselines = nitrogen_model_analysis.baseline_report()





# Carbon

x_cal = load_fp(base_path + "ciso/x_cal.xlsx")
x_test = load_fp(base_path + "ciso/x_test.xlsx")
y_cal = load_fp(base_path + "ciso/y_cal.xlsx")
y_test = load_fp(base_path + "ciso/y_test.xlsx")


ciso_model = regression_model_builder(x_cal = x_cal, x_test = x_test, y_cal = y_cal, y_test = y_test)
ciso_model.generate_xblock()
ciso_model.regression_fit(criteria = 'R2')
ciso_model.reduce_xblock(percentage = 0.02)
ciso_model.predictive_model(ml_type='rf')

ciso_results = ciso_model.results

ciso_model_analysis = visualizer(ciso_model)
ciso_model_analysis.plot_baselines(x = 1, save = False)
ciso_model_analysis.regression_plot(save = False)
ciso_baselines = ciso_model_analysis.baseline_report()





# Cation Exchange Capacity

x_cal = load_fp(base_path + "cec/x_cal.xlsx")
x_test = load_fp(base_path + "cec/x_test.xlsx")
y_cal = load_fp(base_path + "cec/y_cal.xlsx")
y_test = load_fp(base_path + "cec/y_test.xlsx")


cec_model = regression_model_builder(x_cal = x_cal, x_test = x_test, y_cal = y_cal, y_test = y_test)
cec_model.generate_xblock()
cec_model.regression_fit(criteria = 'R2')
cec_model.reduce_xblock(percentage = 0.02)
cec_model.predictive_model(ml_type='rf')


cec_results = cec_model.results

cec_model_analysis = visualizer(cec_model)
cec_model_analysis.plot_baselines(x = 1, save = False)
cec_model_analysis.regression_plot(save = False)
cec_baselines = cec_model_analysis.baseline_report()




