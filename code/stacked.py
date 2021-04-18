#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:11:27 2021

@author: alessandro
"""
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import tools_
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.decomposition import PCA
from time import time
import numpy as np
from sklearn.model_selection import GridSearchCV

X_train, y_train, X_test, y_test = tools_.load_dataset()

# =============================================================================
#  definition of the Base-Models of the stacking classifier
# =============================================================================
random_state = 42

# LogisticRegression
# lr_classifier = LogisticRegression(solver='saga',
#                                     max_iter=8000,
#                                     random_state=42,
#                                     multi_class='multinomial',
#                                     penalty='l1',
#                                     C=1,) 
# estimators = [('scaler', StandardScaler()), 
#               ('lr', lr_classifier)]
# lr_pipe = Pipeline(estimators,verbose=1)

# ExtraTreesClassifier
et_classifier = ExtraTreesClassifier(n_estimators=800,
                                     random_state=random_state,
                                     min_samples_leaf=2)
estimators = [('scaler', StandardScaler()), 
              ('et', et_classifier)]
et_pipe = Pipeline(estimators,verbose=1)

# LinearSVC
lin_svc = LinearSVC(C=0.2, tol=0.00005) 
estimators = [('scaler', StandardScaler()), 
              ('lin_svc', lin_svc)]
lin_svc_pipe = Pipeline(estimators,verbose=1)


# KNeighborsClassifier
estimators = [('scaler', StandardScaler()), 
              ('pca', PCA(random_state=random_state, n_components=266,)),
              ('knn', KNeighborsClassifier(14))] 

knn_pipe = Pipeline(estimators,verbose=1)


# =============================================================================
#  StackingClassifier cross validation tuning parameters
# =============================================================================


estimators = [('svc', lin_svc_pipe),
            ('knn',knn_pipe),
            ('et',et_pipe)] 

final_estimator = LogisticRegression(solver='saga',
                                    max_iter=8000,
                                    random_state=42,
                                    multi_class='multinomial',)

SC = StackingClassifier(estimators=estimators, 
                        final_estimator=final_estimator,
                        verbose=True)  

# rng = np.random.RandomState(42) #to have the same random numbers
# C = rng.uniform(0.5,20,8)
C = [0.0001, 0.001, 0.01, 0.1, 1]
parameters = {'final_estimator__C' : C, 
              'final_estimator__penalty' : ['l2','l1']}

stacked_rs = GridSearchCV(SC, param_grid = parameters,
                                      cv = 5, verbose=1,scoring= 'accuracy', 
                                      n_jobs=-1)

tic = time()
print("Start training")
stacked_rs.fit(X_train, y_train)
toc = time()
print("Trained in "+str(toc-tic)+" seconds")

y_pred = stacked_rs.predict(X_test)
SC_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using stacked classifier:", SC_accuracy)
# Accuracy using Linear SVM: 0.9650492025788938 # NOT FIND ANYMORE
dump(stacked_rs, 'fitted_model/stacked_grid.joblib')
# stacked_rs = load('fitted_model/stacked_rs.joblib')

# =============================================================================
#  StackingClassifier
# =============================================================================

estimators = [('svc', lin_svc_pipe),
            ('knn',knn_pipe),
            ('et',et_pipe)] 

SC = StackingClassifier(estimators=estimators, 
                        final_estimator=LogisticRegression(),
                        verbose=True)  

tic = time()
print("Start training")
SC.fit(X_train, y_train)
toc = time()
print("Trained in "+str(toc-tic)+" seconds")

y_pred = SC.predict(X_test)
SC_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using Linear SVM:", SC_accuracy)

dump(SC, 'SC.joblib')
