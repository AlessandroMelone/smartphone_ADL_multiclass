#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 12:04:17 2021

@author: alessandro

The goal of this file is to know which method to reduce the number of 
components allow a less reduction of the accuracy and what are the 
corresponding number of components of the reduced method.

To assess the quality of the feature reducer method is used the ExtraTrees 
estimator, this choice is due to avoid an estimator that is sensible to its
hyperparameters, i.e. the conclusion obtained would change by changing the
used hyperparameters. Otherwise a cross-validation to tune the hyperparameters
of the estimator should be done (really computationally demanding). 

"""
import tools_

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from time import time
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.ensemble import ExtraTreesClassifier

X_train, y_train, X_test, y_test = tools_.load_dataset()

random_state= 42

# =============================================================================
#  definition of the Base-Models of the stacking classifier
# =============================================================================
classifier = ExtraTreesClassifier(n_estimators=500,random_state=random_state,
                                     verbose=1,)
# note that ExtraTreesClassifier has bootstrap=False as default, so we use all the 
# training set, this choice not increase the variance because here the feature to 
# perform the split are chosen as the optimal element among a random generated set 

# =============================================================================
#  pipeline
# =============================================================================

## feature selectors definition
#are considered the best parameters obtained by cross validation in the other files
rf_classifier = RandomForestClassifier(verbose=1, criterion='gini',
                                       n_estimators=50,
                                       min_samples_leaf=3,
                                       random_state=random_state)

lin_svc = LinearSVC(C=0.2, tol=5e-05,penalty="l1",dual=False)

## reduction methods parameters
# threshold_l1 = [1e-5,  1e-4,  1e-3,  1e-2] #SCV

# threshold_rm_tree = [0.2, 0.5, 0.8, 1.1] #rm_tree

max_features = [100, 200, 300, 400]

# in this case represent a scaling factor of the mean 
# scaling factor (e.g., “0.2*mean”)

n_components = [100, 200, 300, 400] #PCA

## definition method dictionary

pca_dict = {
   "name_method": "PCA",
   "reduce_dim_method": PCA(random_state=random_state),
    "name_dim_param": "n_components",
    "dim_cv_list": n_components,
}
l1_dict = {
   "name_method": "SVC",
   "reduce_dim_method": SelectFromModel(lin_svc),
   # "name_dim_param": "threshold",
   # "dim_cv_list": threshold_l1
   "name_dim_param": "max_features",
   "dim_cv_list" : max_features
}
rm_tree_dict = {
   "name_method": "random tree",
   "reduce_dim_method": SelectFromModel(rf_classifier),
   # "name_dim_param": "threshold",
   # "dim_cv_list": threshold_rm_tree
   "name_dim_param": "max_features",
   "dim_cv_list" : max_features
   
}

# method_list = [l1_dict,rm_tree_dict] 
method_list = [pca_dict, l1_dict,rm_tree_dict] 
#list ofdictionaries, each dict is related with one reduction method

## fit models
grid_cv = []
grid_accuracy = []

for i,method in enumerate(method_list):
    print("CV of the method",method["name_method"]," number "+str(i+1)+" of "
          +str(len(method_list)))
    
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', method["reduce_dim_method"]),
    ('classify', classifier) #use the same classifier
    ], verbose=1)
    
    parameters = {'reduce_dim__'+method["name_dim_param"] : 
                  method["dim_cv_list"]}
    grid_cv.append( GridSearchCV(pipe, param_grid=parameters, 
                              n_jobs=-1,verbose=1))
    
    tic = time()
    print("Start GridSearchCV")
    grid_cv[i].fit(X_train, y_train)
    toc = time()
    print("Trained in "+str(toc-tic)+" seconds")
    
    y_pred = grid_cv[i].predict(X_test)
    # grid_accuracy.append(1)
    grid_accuracy.append(accuracy_score(y_true = y_test, y_pred = y_pred))                         
    print("Accuracy using "+method["name_method"]+": "+str(grid_accuracy[i]))
    dump(grid_cv[i], "fitted_model/dim_red_"+method["name_method"]+".joblib")



