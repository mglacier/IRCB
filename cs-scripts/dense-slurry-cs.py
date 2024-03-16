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
This case study uses xgb to generate regression predictions for nuclear waste slurries 
"""


import sys
sys.path.append('C:\\Users\\spect\\Desktop\\PatternWave\\IRCB - Public\\IRCB')     ################## Change this line for your system
from reg_builder import load_fp
from reg_builder import regression_model_builder, visualizer


base_path = "C:/Users/spect/Desktop/PatternWave/IRCB - Public/IRCB/cs-data/dense-slurry/" ################## Change this line for your system
x_comb, y_comb = load_fp(base_path + "Raman_x_block.csv"), load_fp(base_path + "Raman_y_block.csv")



nuc_solid_model = regression_model_builder(x_comb = x_comb, y_comb = y_comb)
nuc_solid_model.generate_xblock(max_range = 10000)
nuc_solid_model.split_kfold(n_splits = 10)
nuc_solid_model.regression_fit(criteria='R2')
nuc_solid_model.reduce_xblock(percentage = .001)
nuc_solid_model.opt_xgb_randomize(num_iterations=100)
nuc_solid_model.reduce_xblock(percentage = .02)
nuc_solid_model.predictive_model(ml_type='xgb')

results = nuc_solid_model.results

kyanite = visualizer(nuc_solid_model, analyte = 0, fold = 0)
kyanite.plot_baselines(x = 1, save = False)
kyanite.regression_plot(save = False)
kyanite_baselines = kyanite.baseline_report()

wollastonite = visualizer(nuc_solid_model, analyte = 1, fold = 0)
wollastonite.plot_baselines(x = 1, save = False)
wollastonite.regression_plot(save = False)
wollastonite_baselines = wollastonite.baseline_report()

olivine = visualizer(nuc_solid_model, analyte = 2, fold = 0)
olivine.plot_baselines(x = 1, save = False)
olivine.regression_plot(save = False)
olivine_baselines = olivine.baseline_report()

silica = visualizer(nuc_solid_model, analyte = 3, fold = 0)
silica.plot_baselines(x = 1, save = False)
silica.regression_plot(save = False)
silica_baselines = silica.baseline_report()

zircon = visualizer(nuc_solid_model, analyte = 4, fold = 0)
zircon.plot_baselines(x = 1, save = False)
zircon.regression_plot(save = False)
zircon_baselines = zircon.baseline_report()



