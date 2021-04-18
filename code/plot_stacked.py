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
# from sklearn.metrics import confusion_matrix

X_train, y_train, X_test, y_test = tools_.load_dataset()

# =============================================================================
#  STACKED
# =============================================================================

# SC = load('fitted_model/SC.joblib')
stacked_grid = load('fitted_model/stacked_grid.joblib')

tools_.plot_grid_search(stacked_grid, 'final_estimator__C', 
                        'final_estimator__penalty', 
                        'C', 'Penalty')

# y_pred = stacked_rs.predict(X_test)
# SC_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
# print("Accuracy using stacked classifier:", SC_accuracy)

tic = time()
print("Start testing")
y_pred = stacked_grid.predict(X_test)
toc = time()
print("Testing in "+str(toc-tic)+" seconds")
stacked_grid_test_time = toc-tic

stacked_grid_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using stacked classifier:", stacked_grid_accuracy)

best_penalty = stacked_grid.best_params_["final_estimator__penalty"]
best_C = stacked_grid.best_params_["final_estimator__C"]
print("Best set of parameters:")
print(f"penalty: {best_penalty}      C: {best_C}")

clf = stacked_grid
temp = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                  pd.DataFrame(clf.cv_results_["mean_test_score"], 
                               columns=["mean_test_score"]),
                  pd.DataFrame(clf.cv_results_["std_test_score"], 
                               columns=["std_test_score"])],axis=1)
temp = temp.rename(columns={'lin_svc__C': 'C'})
print(temp)


tools_.plot_confusion_matrices(name_method = "", 
                                name_parameter = "Stacked Classifier", 
                                y_test = y_test, 
                                y_pred = y_pred)

