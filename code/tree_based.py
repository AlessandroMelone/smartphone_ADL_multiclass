#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:08:52 2021

@author: alessandro
"""
# TODO:
#     use ExtraTreesClassifier
#     do this https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py
#     use svc as shown here https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel
#     finally do PCA https://scikit-learn.org/stable/modules/decomposition.html#pca 

import tools_
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import graphviz 

# rf_classifier_rs = load('fitted_model/rf_classifier.joblib') 

X_train, y_train, X_test, y_test = tools_.load_dataset()


#use gini index or RSS

# Decision tree model with Hyperparameter tuning and cross validation
parameters = {'max_depth':np.arange(2,10,2)}
dt_classifier = tree.DecisionTreeClassifier(criterion='gini', verbose=True)
dt_classifier_rs = RandomizedSearchCV(dt_classifier,param_distributions=parameters,
                                      random_state = 42)
dt_classifier_rs.fit(X_train, y_train)
y_pred = dt_classifier_rs.predict(X_test)
dt_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using Decision tree:", dt_accuracy)
print("Best parameter:", dt_classifier_rs.best_params_)

## getting best estimators

# =============================================================================
#  view tree
# =============================================================================
labels=['LAYING', 'SITTING','STANDING','WALKING','WALKING_DS','WALKING_US']

clf = dt_classifier_rs.best_estimator_
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("tree") 
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=X_train.columns.to_list(),  
                     class_names=labels,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.view()


# =============================================================================
# GridSearch
# =============================================================================

RandomForest = True
if RandomForest:
    forest = RandomForestClassifier(criterion='gini',random_state=42,
                                    n_jobs=-1)
else:
    forest = ExtraTreesClassifier(criterion='gini',random_state=42,
                                  n_jobs=-1)
    
estimators = [('scaler', StandardScaler()), 
              ('forest', forest)]
pipe = Pipeline(estimators)

parameters = {'forest__min_samples_leaf':[1,2,3],
               'forest__n_estimators' :[20,30, 50,100,200,400,600,800]}
forest_grid = GridSearchCV(pipe, param_grid=parameters, n_jobs=-1, 
                           verbose=1)
#plot result in function of C

forest_grid.fit(X_train, y_train)

y_pred = forest_grid.predict(X_test)
forest_grid_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy:", forest_grid_accuracy)

if RandomForest:
    dump(forest_grid, 'fitted_model/RandomForest_grid.joblib') 
else:
    dump(forest_grid, 'fitted_model/ExtraTrees_grid.joblib') 



# =============================================================================
# Random Forest model using Hyperparameter tuning and cross validation
# =============================================================================

params = {'n_estimators': 100+np.arange(20,101,10), 'max_depth':np.arange(2,16,2)}
# the number of estimator should be always the greater we can, is silly to find the 
# optimal case

rf_classifier = RandomForestClassifier(verbose=1, criterion='gini')
# rf_classifier_rs = RandomizedSearchCV(rf_classifier, param_distributions=params,
#                                       random_state=42,verbose=1,n_iter=12,cv = 5)

# let go depth the trees to have low bias despite the high variance
rf_classifier = RandomForestClassifier(verbose=1, criterion='gini',
                                        n_estimators=800,random_state=42,max_features=10)


# rf_classifier_rs.fit(X_train, y_train)

# y_pred = rf_classifier_rs.predict(X_test)
# rf_accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy using Random Forest:", rf_accuracy)

# dump(rf_classifier, 'fitted_model/rf_classifier.joblib') 


# =============================================================================
#  extra trees
# =============================================================================

# ExtraTreesClassifier
et_classifier = ExtraTreesClassifier(n_estimators=800,random_state=42,
                                     verbose=1,)
# note that ExtraTreesClassifier has bootstrap=False as default, so we use all the 
# training set, this choice not increase the variance because here the feature to 
# perform the split are chosen as the optimal element among a random generated set 

et_classifier.fit(X_train, y_train)

y_pred = et_classifier.predict(X_test)
et_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy using Extra Trees:", et_accuracy)

et_classifier = load('fitted_model/et_classifier.joblib') 
# dump(et_classifier, 'fitted_model/et_classifier.joblib') 


# max_feat =10, accuracy = 0.9436715303698676 
# max_feat =default, accuracy = 0.9402782490668476

# =============================================================================
# pipeline
# =============================================================================


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
et_classifier = ExtraTreesClassifier(n_estimators=800,random_state=42,
                                     verbose=1,)

estimators = [('scaler', StandardScaler()), 
              ('et', et_classifier)]
pipe = Pipeline(estimators)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
et_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy using Extra Trees:", et_accuracy)



# =============================================================================
#  features importance visualization
# =============================================================================
# not reported in the report because the obtained plot is not useful for 
# the goal of the project, but is still present here hence it could be used
# in a case of study in which the goal is also to do inference

forest = pipe['et']
# forest = rf_classifier
# forest = et_classifier

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0) 
#to see if the feature_importances_ how much varies within the trees that composes
#the random forest

indices = np.argsort(importances)[::-1]
# [::-1] do a flipping: the first element become the last, the last become the first, etc...

#list ordered of the features names with higher value
X_train.columns.to_numpy()[indices] 

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show() #repeat training with more trees to reduce the variance!
# try with ExtraTreesClassifier

# from the plot we see a great variance 
