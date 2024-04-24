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
    
    WARNING: PRESSING PLAY WILL DEFAULT TO ONE COMPUTATIONAL CORE

"""



"""
This case study uses ELR to generate regression predictions for a 3 component mixture of propofol, hdipba, and 2-ip
The external test set is predefined
"""



import sys
sys.path.append('C:\\Users\\spect\\Desktop\\PatternWave\\IRCB - Public\\IRCB')  # Change for your System
from reg_builder import load_fp
from reg_builder import regression_model_builder, visualizer


base_path = "C:/Users/spect/Desktop/PatternWave/IRCB - Public/IRCB/cs-data/propofol-3comp/" # Change for your System



x_cal = load_fp(base_path + "x_cal.xlsx")
x_test = load_fp(base_path + "x_test.xlsx")
y_cal = load_fp(base_path + "y_cal.xlsx")
y_test = load_fp(base_path + "y_test.xlsx")


model = regression_model_builder(x_cal = x_cal, x_test = x_test, y_cal = y_cal, y_test = y_test)
model.generate_xblock()
model.regression_fit(criteria = 'R2')
model.reduce_xblock(percentage = 0.02)


model.predictive_model(ml_type = 'elr')
results = model.results



############ Propofol, HDIPBA, 2-IP #############

propofol_analysis = visualizer(model, analyte = 0)
propofol_analysis.plot_baselines(x = 8, x_limits=[700, 2000], y_limits = [-0.01,0.4], save = False)
propofol_analysis.regression_plot(save = False)
prop_baselines = propofol_analysis.baseline_report()


hdipba_analysis = visualizer(model, analyte = 1)
hdipba_analysis.plot_baselines(x = 8, x_limits=[1600, 1800], y_limits = [-0.01,0.06], save = False)
hdipba_analysis.regression_plot(save = False)
hdipba_baselines = hdipba_analysis.baseline_report()

ip_analysis = visualizer(model, analyte = 2)
ip_analysis.plot_baselines(x = 8, x_limits=[800, 1600], y_limits = [0.0,0.3], save = False)
ip_analysis.regression_plot(save = False)
ip_baselines = ip_analysis.baseline_report()


