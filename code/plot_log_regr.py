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
import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

X_train, y_train, X_test, y_test = tools_.load_dataset()

# =============================================================================
#  LOGISTIC REGRESSION
# =============================================================================

lr_class_rs_l1l2 = load('fitted_model/lr_classifier_grid_l1l2.joblib') 
lr_class_rs_el = load('fitted_model/lr_classifier_grid_el.joblib') 


tools_.plot_grid_search(lr_class_rs_l1l2, 'lr__C', 'lr__penalty', 
                        'C', 'Penalty')
tools_.plot_grid_search(lr_class_rs_el, 'lr__C', 'lr__l1_ratio', 
                        'C', 'l1_ratio')



lr_list = [["l1", lr_class_rs_l1l2],["elasticnet", lr_class_rs_el]]
for name_penalty , lr_classifier in lr_list:
    print("\n")
    print(f"++++++    {name_penalty}    ++++++")
    # testing classifier
    y_pred = lr_classifier.predict(X_test)
    lr_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
    print("Accuracy using Logistic Regression:", lr_accuracy)
    
    best_penalty = lr_classifier.best_params_["lr__penalty"]
    best_C = lr_classifier.best_params_["lr__C"]
    print("Best set of parameters:")
    print(f"penalty: {best_penalty}      C: {best_C}")
    
    clf = lr_classifier 
    temp = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                      pd.DataFrame(clf.cv_results_["mean_test_score"], 
                                   columns=["mean_test_score"]),
                      pd.DataFrame(clf.cv_results_["std_test_score"], 
                                   columns=["std_test_score"])],axis=1)
    temp = temp.rename(columns={'lr__penalty': 'penalty','lr__C': 'C',
                                'lr__l1_ratio': 'l1 ratio'})
    print(temp)

    
    tools_.plot_confusion_matrices(name_method = "Logistic Regression", 
                                    name_parameter = name_penalty, 
                                    y_test = y_test, 
                                    y_pred = y_pred)
    
    coef_l1_LR = lr_classifier.best_estimator_.steps[1][1].coef_.ravel()
    sparsity_l1_LR = np.mean(np.abs(coef_l1_LR) <= 10e-5) * 100
    print("{:<40} {:.2f}%".format(f"Sparsity with {name_penalty} penalty:", sparsity_l1_LR))
    #problably is not so much sparse because the algorithm do not converge, 
    #should use more steps
    
    plt.figure(figsize=(3.5,3.5))
    plt.title(f"Coeffient best estimator with {name_penalty} penalty", fontsize=15, fontweight='bold')
    Z = lr_classifier.best_estimator_.steps[1][1].coef_
    plt.pcolor(np.absolute(Z),cmap = 'Greys')
    plt.colorbar()
    plt.show()

