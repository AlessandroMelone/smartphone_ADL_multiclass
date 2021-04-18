#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:16:02 2021

@author: alessandro

Plot of the result obtained by the models trained in the others file,
the models are loaded through the function "load", the plots are obtained
thanks functions implemented in "tools_.py"
"""
from time import time
from sklearn.metrics import accuracy_score
from joblib import load
import pandas as pd
import tools_

X_train, y_train, X_test, y_test = tools_.load_dataset()

reduced_stacked_grid = load('fitted_model/stacked_red_feat_grid.joblib')

tic = time()
# print("Start testing")
y_pred = reduced_stacked_grid.predict(X_test)
toc = time()
# print("Testing in "+str(toc-tic)+" seconds")
reduced_stacked_test_time = toc-tic


tools_.plot_confusion_matrices(name_method = "", 
                                name_parameter = "Reduced features Stacked Classifier", 
                                y_test = y_test, 
                                y_pred = y_pred)

tools_.plot_grid_search(reduced_stacked_grid, 'classify__final_estimator__C', 
                        'classify__final_estimator__penalty', 
                        'C', 'Penalty')


reduced_stacked_grid_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using reducing features stacked classifier:", 
      reduced_stacked_grid_accuracy)

best_penalty = reduced_stacked_grid.best_params_["classify__final_estimator__penalty"]
best_C = reduced_stacked_grid.best_params_["classify__final_estimator__C"]
print("Best set of parameters:")
print(f"penalty: {best_penalty}      C: {best_C}")

clf = reduced_stacked_grid
temp = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                  pd.DataFrame(clf.cv_results_["mean_test_score"], 
                               columns=["mean_test_score"]),
                  pd.DataFrame(clf.cv_results_["std_test_score"], 
                               columns=["std_test_score"])],axis=1)
temp = temp.rename(columns={'classify__final_estimator__C': 'C'})
temp = temp.rename(columns={'classify__final_estimator__penalty': 'penalty'})
print(temp)

# perc_reduced = 100-reduced_stacked_test_time/stacked_grid_test_time*100
# np.round(perc_reduced,4)
# print(f"The prediction time has been reduced of the {perc_reduced} %")

