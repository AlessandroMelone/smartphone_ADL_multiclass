#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:11:27 2021

@author: alessandro

In this file is used the estimator that has the highest accuracy in the case
of all feature (the stacked estimator) but using the best feature reduction 
method finded in the features_importance.py file, the cross-validation to tune 
the estimator is not performed for computational reasons.

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
from joblib import dump
from sklearn.decomposition import PCA
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

X_train, y_train, X_test, y_test = tools_.load_dataset()

# =============================================================================
#  definition of the Base-Models of the stacking classifier
# =============================================================================
# same Base-Models used in the full features case

random_state = 42

# ExtraTreesClassifier
et_classifier = ExtraTreesClassifier(n_estimators=800,
                                     random_state=random_state,
                                     min_samples_leaf=2)

# LinearSVC
lin_svc = LinearSVC(C=0.2, tol=0.00005)

# KNeighborsClassifier
estimators = [('pca', PCA(random_state=random_state, n_components=266,)),
              ('knn', KNeighborsClassifier(14))] 

knn_pipe = Pipeline(estimators,verbose=1)
#knn use PCA to have the same configuration of the all features case, so we
# can use the optimal parameters found in that case


# =============================================================================
#  StackingClassifier
# =============================================================================

estimators = [('svc', lin_svc),
            ('knn',knn_pipe),
            ('et',et_classifier)] 

# best final_estimator obtained through the CV tuning of the stacked classifier
# with full number of features 
final_estimator = LogisticRegression(solver='saga',
                                    max_iter=8000,
                                    random_state=42,
                                    multi_class='multinomial',
                                    C = 0.001,
                                    penalty = "l2"
                                    )
SC = StackingClassifier(estimators=estimators, 
                        final_estimator=final_estimator,
                        verbose=True, n_jobs=-1)  

# =============================================================================
#  model selection estimator
# =============================================================================
lin_svc = LinearSVC(C=0.2, tol=5e-05,penalty="l1",dual=False)

# =============================================================================
# pipeline definition
# =============================================================================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('reduce_dim', SelectFromModel(lin_svc, max_features=300)),
    ('classify', SC) #use the same classifier
    ], verbose=1)
    
C = [0.0001, 0.001, 0.01, 0.1, 1]
parameters = {'classify__final_estimator__C' : C, 
              'classify__final_estimator__penalty' : ['l2','l1']}

stacked_rs = GridSearchCV(pipe, param_grid = parameters,
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
dump(stacked_rs, 'fitted_model/stacked_red_feat_grid.joblib')


