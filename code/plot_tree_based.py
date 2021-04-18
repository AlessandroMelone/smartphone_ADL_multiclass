#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 21:16:02 2021

@author: alessandro

Plot of the result obtained by the models trained in the others file,
the models are loaded through the function "load", the plots are obtained
thanks functions implemented in "tools_.py"
"""
from sklearn.metrics import accuracy_score
from joblib import load
import pandas as pd
import tools_
# from sklearn.metrics import confusion_matrix

X_train, y_train, X_test, y_test = tools_.load_dataset()

# =============================================================================
# Trees 
# =============================================================================
rf_grid = load('fitted_model/RandomForest_grid.joblib') 
et_grid = load('fitted_model/ExtraTrees_grid.joblib') 
# rf_grid=forest_grid

tools_.plot_grid_search(rf_grid, 'forest__n_estimators', 
                        'forest__min_samples_leaf', 
                        'num. estimator', 'min samples leaf',
                        x_log_scale=False)
tools_.plot_grid_search(et_grid, 'forest__n_estimators', 
                        'forest__min_samples_leaf', 
                        'num. estimator', 'min samples leaf',
                        x_log_scale=False)


forest_list = [["Random Forest", rf_grid],["Extra Trees", et_grid]]
for name_method , forest_classifier in forest_list:
    print("\n")
    print(f"++++++    {name_method}    ++++++")
    # testing classifier
    y_pred = forest_classifier.predict(X_test)
    forest_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
    print("Accuracy using Forest Classifier:", forest_accuracy)
    
    best_samp_leaf = forest_classifier.best_params_["forest__min_samples_leaf"]
    best_n_estim = forest_classifier.best_params_["forest__n_estimators"]
    print("Best set of parameters:")
    print(f"min samples leaf: {best_samp_leaf}      num. estimator: {best_n_estim}")
    
    clf = forest_classifier 
    temp = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                      pd.DataFrame(clf.cv_results_["mean_test_score"], 
                                   columns=["mean_test_score"]),
                      pd.DataFrame(clf.cv_results_["std_test_score"], 
                                   columns=["std_test_score"])],axis=1)
    temp = temp.rename(columns={'forest__min_samples_leaf': 'min_samples_leaf',
                                'forest__n_estimators': 'n_estimators'})
    print(temp)

    
    tools_.plot_confusion_matrices(name_method = "", 
                                    name_parameter = name_method, 
                                    y_test = y_test, 
                                    y_pred = y_pred)
