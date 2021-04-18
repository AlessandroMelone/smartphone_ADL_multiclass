#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:11:27 2021

@author: alessandro
"""
import tools_
from joblib import dump
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# from tempfile import mkdtemp
# from shutil import rmtree
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

X_train, y_train, X_test, y_test = tools_.load_dataset()

# =============================================================================
# define pipeline svm
# =============================================================================
estimators = [('scaler', StandardScaler()), 
              ('svc', SVC(kernel = 'poly',decision_function_shape = 'ovr',))]
pipe = Pipeline(estimators)

#define RandomizedSearchCV
rng = np.random.RandomState(42) #to have the same random numbers
C = rng.uniform(0.5,20,12)
gamma = rng.uniform(0.001,5,12)

parameters = {'svc__C' : C, 'svc__gamma' : gamma}

kernel_svm_rs = RandomizedSearchCV(pipe, param_distributions = parameters,
                                    random_state = 42, verbose=1,n_iter =15)

kernel_svm_rs.fit(X_train, y_train)

# kernel_svm_rs = load('fitted_model/kernel_svm_rs.joblib') 
y_pred = kernel_svm_rs.best_estimator_.predict(X_test)
kernel_svm_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using Kernel SVM:", kernel_svm_accuracy)

dump(kernel_svm_rs, 'fitted_model/kernel_svm_rs.joblib') 

# =============================================================================
# poly 
# =============================================================================

estimators = [('scaler', StandardScaler()), 
              ('svc', SVC(kernel='poly',decision_function_shape = 'ovr',))]
pipe = Pipeline(estimators, verbose=1)

#define RandomizedSearchCV
rng = np.random.RandomState(42) #to have the same random numbers
C = rng.uniform(0.5,20,12)
gamma = rng.uniform(0.001,5,12)

parameters = {'svc__C' : C, 'svc__gamma' : gamma, 'svc__degree' : [2, 3, 4],
              'svc__coef0' : [0, 0.2, 0.4]}

kernel_svm_rs = RandomizedSearchCV(pipe, param_distributions = parameters,njobs=-1,
                                    random_state = 42, verbose=1,n_iter =15)

kernel_svm_rs.fit(X_train, y_train)

# kernel_svm_rs = load('fitted_model/kernel_svm_rs_poly.joblib') 
y_pred = kernel_svm_rs.best_estimator_.predict(X_test)
kernel_svm_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using Kernel SVM:", kernel_svm_accuracy)

dump(kernel_svm_rs, 'fitted_model/kernel_svm_rs_poly.joblib') 




# =============================================================================
#  Linear SVC no pipeline
# =============================================================================

# Linear SVM model with Hyperparameter tuning and cross validation

# parameters = {'C': np.arange(1,12,2)}
parameters = {'C': np.arange(0.01,1,1),}
lin_svm = LinearSVC(tol = 0.00005)
lin_svm_grid = GridSearchCV(lin_svm, param_distributions = parameters, verbose=1)
lin_svm_grid.fit(X_train, y_train)
y_pred = lin_svm_grid.predict(X_test)
lin_svm_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using Linear SVM:", lin_svm_accuracy)

dump(lin_svm_grid, 'fitted_model/lin_svm_rs.joblib') 


# =============================================================================
#  linear svc pipeline
# =============================================================================
l1 = True
if l1:
    lin_svc = LinearSVC(tol=0.00005,penalty="l1",dual=False)
else:
    lin_svc = LinearSVC(tol=0.00005,penalty="l2")
    
estimators = [('scaler', StandardScaler()), 
              ('lin_svc', lin_svc)]
pipe = Pipeline(estimators)

parameters = {'lin_svc__C':[0.01, 0.05, 0.1, 0.2, 0.8, 1.5, 3],}
              # 'lin_svc__penalty' : ['l2','l1']}
lin_svc_grid = GridSearchCV(pipe, param_grid=parameters, n_jobs=-1, 
                           verbose=1)
#plot result in function of C

lin_svc_grid.fit(X_train, y_train)

y_pred = lin_svc_grid.predict(X_test)
lin_svc_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using Linear SVM:", lin_svc_accuracy)

if l1:
    dump(lin_svc_grid, 'fitted_model/lin_svc_grid_l1.joblib') 
else:
    dump(lin_svc_grid, 'fitted_model/lin_svc_grid_l2.joblib') 


# dump(lin_svc_grid, 'fitted_model/lr_svc_grid_l1.joblib') 
# dump(lin_svc_grid, 'fitted_model/lr_svc_grid_l2.joblib') 


# lin_svc_grid = load('fitted_model/lr_svc_grid_l1.joblib') 




