
DISCLAIMER: This code is provided for academic and educational purposes only.
Commercial use, redistribution, or any other form of reproduction for profit
without explicit written consent from the author is strictly prohibited.

The author of this code is not responsible for any misuse or unauthorized use,
and assumes no liability for any consequences resulting from the use of this code.

If you wish to use this code for commercial purposes or have any inquiries
regarding its use, please contact the author Matthew Glace at glacemk@vcu.edu.

Created by Matthew Glace, 2023


INSTRUCTIONS:

This is a python package for running iterative regression of corrective baselines (IRCB) 
for spectroscopic model development. Three case studies are included.

In order to run the case studies, filepaths must be adjusted for the following files (at the top)
1) reg_builder.py
2) ml_models.py
3) the case study file being run

A list of dependencies can be found in requirements.txt 

Python v 3.9.18

Run a case study script to begin. 
CS 1 - propofol-3comp-cs 
CS 2 - soil-nir-cs
CS 3 - dense-slurry-cs (Most computationally expensive)


Description of Each File:
parallel_computate.py     - generates the X_Transform matrix using all avaliable computational cores
sorting.py                - assigns R^2 for each X_Transform element
stats.py                  - case study statistics
ml_models.py              - machine learning models for ELR, random forest, XGB
hyp_opt.py		  - hyperparameter optimization function for XGB
reg_builder.py            - Main Classes for building regression models, visualizing results and loading data
requirements.txt          - dependencies



