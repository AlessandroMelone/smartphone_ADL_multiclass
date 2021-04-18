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

## load models
max_features = [100, 200, 300, 400]
n_components = [100, 200, 300, 400] #PCA

pca_dict = {
   "name_method": "PCA",
    "name_dim_param": "n_components",
    "dim_cv_list": n_components,
}
l1_dict = {
   "name_method": "SVC",
   "name_dim_param": "max_features",
   "dim_cv_list" : max_features
}
rm_tree_dict = {
   "name_method": "random tree",
   "name_dim_param": "max_features",
   "dim_cv_list" : max_features
   
}

method_list = [pca_dict, l1_dict,rm_tree_dict] 
grid_cv = [None]*len(method_list)
grid_accuracy = [None]*len(method_list)

for i,method in enumerate(method_list):
    grid_cv[i] = load("fitted_model/dim_red_"+method["name_method"]+".joblib")
    y_pred = grid_cv[i].predict(X_test)
    grid_accuracy[i] = accuracy_score(y_true = y_test, y_pred = y_pred)
    tools_.plot_confusion_matrices(name_method = "Extra Trees with reduction", 
                                    name_parameter = method["name_method"], 
                                    y_test = y_test, 
                                    y_pred = y_pred)
                         

print("\n")
for i,method in enumerate(method_list):    
    name_method = method["name_method"]
    print(f"++++++    {name_method.upper()}    ++++++")
    print(f"Best accuracy using {name_method}: {grid_accuracy[i]}")
    best_reduce_dim = grid_cv[i].best_params_['reduce_dim__'+method["name_dim_param"]]
    print(f"Best reduced number of features: {best_reduce_dim}")
    grid_cv[i].best_params_['reduce_dim__'+method["name_dim_param"]]
    clf = grid_cv[i]
    temp = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                      pd.DataFrame(clf.cv_results_["mean_test_score"], 
                                   columns=["mean_test_score"]),
                      pd.DataFrame(clf.cv_results_["std_test_score"], 
                                   columns=["std_test_score"])],axis=1)
    temp = temp.rename(columns={'reduce_dim__max_features': 
                                'max number of features'})
    print(temp)
    print("\n\n")

name_param_1 = "number features"
name_param_2 = "Features selector"
grid_param_1 = [100, 200, 300, 400] 
grid_param_2 =["PCA", "SVC", "random tree"]
tools_.plot_grid_search_1param(grid_cv, grid_param_1, grid_param_2,
                               name_param_1, name_param_2)

# plot logistic regression Random search could be done with the plot_grid_search_2params 
# method

