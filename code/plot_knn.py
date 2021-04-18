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
import numpy as np
# from sklearn.metrics import confusion_matrix

X_train, y_train, X_test, y_test = tools_.load_dataset()
    
# =============================================================================
#  KNN 
# =============================================================================
pca_knn_rs = load('fitted_model/pca_knn_rs.joblib') 
knn_grid = load('fitted_model/knn_grid.joblib') 

knn_list = [ ["PCA and KNN", pca_knn_rs],
             ["KNN", knn_grid]]

for name,clf in knn_list:    
    print(f"++++++    {name}    ++++++")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
    print(f"Accuracy using {name}:", accuracy)
    tools_.plot_confusion_matrices(name_method = "", 
                            name_parameter = name, 
                            y_test = y_test, 
                            y_pred = y_pred)
    

    if(name == "PCA and KNN"):
        print("Best model number of neighbors "+
              str(clf.best_params_['knn__n_neighbors']))
        print("Best model number of PCA components "+
              str(clf.best_params_['pca__n_components']))
        print('\n')
        print(f"\n CV result of: {name}")
        temp = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                          pd.DataFrame(clf.cv_results_["mean_test_score"],
                                       columns=["mean_test_score"]),
                          pd.DataFrame(clf.cv_results_["std_test_score"],
                                       columns=["std_test_score"])],axis=1)
        temp = temp.rename(columns={'knn__n_neighbors': 'number neighb.'})
        temp = temp.rename(columns={'pca__n_components': 'PCA number components.'})
        print(temp)
        print("\n")
    else:
        print("Best model number of neighbors "+
              str(clf.best_params_['knn__n_neighbors']))
        print('\n')
        print(f"\n CV result of: {name}")
        temp = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                          pd.DataFrame(clf.cv_results_["mean_test_score"],
                                       columns=["mean_test_score"]),
                          pd.DataFrame(clf.cv_results_["std_test_score"],
                                       columns=["std_test_score"])],axis=1)
        
        temp = temp.rename(columns={'knn__n_neighbors': 'number neighb.'})
        print(temp)
        print("\n")

  
    # best_penalty = lr_classifier.best_params_["lr__penalty"]
    # best_C = lr_classifier.best_params_["lr__C"] best set of parameters
    # print("Best set of parameters:")
    # print(f"penalty: {best_penalty}      C: {best_C}")

name_param_1 = "number of neighbours"
name_param_2 = ""
grid_param_1 = np.arange(2,40,1)
grid_param_2 =[""]
knn_list = [knn_grid]
tools_.plot_grid_search_1param(knn_list, grid_param_1, grid_param_2,
                               name_param_1, name_param_2, labels=False)    


nca_knn = load('fitted_model/nca_knn.joblib')
y_pred = nca_knn.predict(X_test)
nca_knn_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using NeighborhoodComponentsAnalysis and KNN:", nca_knn_accuracy)

