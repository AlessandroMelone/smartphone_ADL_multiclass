#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 23:53:44 2021

@author: alessandro
"""
import tools_
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
import numpy as np
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from joblib import load

# lr_classifier_rs = load('fitted_model/log_regr_classif_multinomial.joblib') 
lr_classifier_rs = load('fitted_model/lr_classifier_rs_l1l2.joblib')

random_state = 42

X_train, y_train, X_test, y_test = tools_.load_dataset()

# =============================================================================
# finding the best l1 or l2
# =============================================================================
#Choose which penalty use: if elasticnet = False the searching will be done  
# with l1 and l2 
elasticnet = True

lr_classifier = LogisticRegression(solver='saga',
                                    max_iter=200,
                                    random_state=42,
                                    multi_class='multinomial')

estimators = [('scaler', StandardScaler()), 
              ('lr', lr_classifier)]

pipe = Pipeline(estimators,verbose=1)

if(elasticnet):
    # rng = np.random.RandomState(42) #to have the same random numbers
    # C = rng.uniform(0.1,1,3)
    C = [0.01, 0.1, 1]
    # l1_ratio = rng.uniform(0.1,1,3)
    l1_ratio = [0.25, 0.5, 0.75]
    parameters = {'lr__C' : C, 
                  'lr__penalty' : ['elasticnet'],  
                  'lr__l1_ratio' : l1_ratio}
else:
    # rng = np.random.RandomState(42) #to have the same random numbers
    # C = rng.uniform(0.1,1,5)
    C = [0.0001, 0.001, 0.01, 0.1, 1]
    parameters = {'lr__C' : C, 
                  'lr__penalty' : ['l2','l1']}

lr_classifier_grid = GridSearchCV(pipe, param_grid = parameters,
                                      cv = 5, verbose=1,scoring= 'accuracy',
                                      n_jobs=-1)



lr_classifier_grid.fit(X_train, y_train)

# knn_rs = load('fitted_model/knn_rs.joblib') 
y_pred = lr_classifier_grid.best_estimator_.predict(X_test)
lr_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using Logisitc Regression:", lr_accuracy)
print("Best estimator parameters: ",lr_classifier_grid.best_params_)

# lr_classifier_rs = load('fitted_model/lr_classifier_rs_pipe.joblib') 
if(elasticnet):
    dump(lr_classifier_grid, 'fitted_model/lr_classifier_grid_el.joblib') 
else:
    dump(lr_classifier_grid, 'fitted_model/lr_classifier_grid_l1l2.joblib') 


# =============================================================================
# No standard scaling
# =============================================================================
parameters = {'C' : uniform(loc=0, scale=4), 
              'penalty' : ['l2','l1','elasticnet'],  
              'l1_ratio' : np.arange(0.2,0.9,0.1)}

lr_classifier = LogisticRegression(solver='saga',
                                    max_iter=200,
                                    random_state=42,
                                    verbose=1,
                                    multi_class='multinomial')
lr_classifier_rs = RandomizedSearchCV(lr_classifier, param_distributions = parameters, \
                                      cv = 5, random_state = 42, verbose=1,\
                                      scoring= 'accuracy', n_iter=12,n_jobs=-1)

lr_classifier_rs.fit(X_train, y_train)
lr_classifier_rs.cv_results_

# save classifier
# dump(lr_classifier_rs, 'fitted_model/log_regr_classif_multinomial.joblib') 
# lr_classifier_rs = load('fitted_model/log_regr_classif_multinomial.joblib') 

# testing classifier
y_pred = lr_classifier_rs.predict(X_test)
lr_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using Logisitc Regression:", lr_accuracy)

print("Best set of parameters:", lr_classifier_rs.best_params_)
print("Best score:", lr_classifier_rs.best_estimator_)
    



# # # =============================================================================
# # #  do more iteration on the optimal case log_regr_classif_multinomial_converg.joblib
# # # =============================================================================

# # lr_classifier_optimal = LogisticRegression(solver='saga',
# #                                            max_iter=8000,
# #                                            random_state=42,
# #                                            verbose=1,
# #                                            multi_class='multinomial',
# #                                            penalty='l1',
# #                                            C=1.8242799368681437,)

# # lr_classifier_optimal.fit(X_train, y_train)  
# # # testing classifier
# # y_pred = lr_classifier_rs.predict(X_test)
# # lr_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
# # print("Accuracy using Logisitc Regression:", lr_accuracy)

# # print("Best set of parameters:", lr_classifier_rs.best_params_)
# # print("Best score:", lr_classifier_rs.best_estimator_)