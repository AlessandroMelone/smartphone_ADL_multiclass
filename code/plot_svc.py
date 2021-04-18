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
#  SVC
# =============================================================================
svc_list = []  
svc_list.append(load('fitted_model/lin_svc_grid_l1.joblib'))
svc_list.append(load('fitted_model/lin_svc_grid_l2.joblib')) 


name_param_1 = "C"
name_param_2 = "Penalty"
grid_param_1 = [0.01, 0.05, 0.1, 0.2, 0.8, 1.5, 3]
grid_param_2 =["l1", "l2"]
tools_.plot_grid_search_1param(svc_list, grid_param_1, grid_param_2,
                               name_param_1, name_param_2)

svc_list = [ ["l1", svc_list[0]],
             ["l2", svc_list[1]] ]

for name,clf in svc_list:
    print(f"\n SVC CV result with penalty: {name}")
    temp = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                      pd.DataFrame(clf.cv_results_["mean_test_score"], 
                                   columns=["mean_test_score"]),
                      pd.DataFrame(clf.cv_results_["std_test_score"], 
                                   columns=["std_test_score"])],axis=1)
    temp = temp.rename(columns={'lin_svc__C': 'C'})
    print(temp)
    
    print(f"Best parameters with penalty: {name}")
    print(f"C : {clf.best_params_['lin_svc__C']}")
    y_pred = clf.predict(X_test)        
    accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
    print(f"Accuracy best model with penalty {name}:", accuracy)

    tools_.plot_confusion_matrices(name_method = "SVC", 
                                name_parameter = name, 
                                y_test = y_test, 
                                y_pred = y_pred)
    
