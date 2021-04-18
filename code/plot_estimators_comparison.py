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
import tools_
import warnings
warnings.filterwarnings("ignore")

X_train, y_train, X_test, y_test = tools_.load_dataset()


print("\n")
# =============================================================================
#  LOGISTIC REGRESSION
# =============================================================================

lr_class_rs_l1l2 = load('fitted_model/lr_classifier_grid_l1l2.joblib') 
lr_class_rs_el = load('fitted_model/lr_classifier_grid_el.joblib') 


print("++++++    LOGISTIC REGRESSION CV BEST MODELS    ++++++")
lr_list = [["l1", lr_class_rs_l1l2],["elasticnet", lr_class_rs_el]]
for name_penalty , lr_classifier in lr_list:
    # testing classifier
    y_pred = lr_classifier.predict(X_test)
    lr_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
    print(f"Accuracy with penalty {name_penalty}:", lr_accuracy)
    
    std = tools_.get_std(lr_classifier)
    print("std_test_score best hyper parameters:", std)
    

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

svc_list = [ ["l1", svc_list[0]],
             ["l2", svc_list[1]] ]

print("++++++   SVC CV BEST MODELS    ++++++")
for name,clf in svc_list:
    y_pred = clf.predict(X_test)        
    accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
    print(f"Accuracy with penalty: {name}:", accuracy)

    std = tools_.get_std(clf)
    print("std_test_score best hyper parameters:", std)
    
    
    
# =============================================================================
#  KNN 
# =============================================================================
pca_knn_rs = load('fitted_model/pca_knn_rs.joblib') 
knn_grid = load('fitted_model/knn_grid.joblib') 

knn_list = [ ["PCA and KNN", pca_knn_rs],
             ["KNN", knn_grid]]

print("++++++   KNN CV BEST MODELS    ++++++")
for name,clf in knn_list:    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
    print(f"Accuracy using {name}:", accuracy)

    std = tools_.get_std(clf)
    print("std_test_score best hyper parameters:", std)



# =============================================================================
# Trees 
# =============================================================================

rf_grid = load('fitted_model/RandomForest_grid.joblib') 
et_grid = load('fitted_model/ExtraTrees_grid.joblib') 


print("++++++    FOREST CLASSIFIERS CV BEST MODELS    ++++++")
forest_list = [["Random Forest", rf_grid],["Extra Trees", et_grid]]
for name_method , forest_classifier in forest_list:
    y_pred = forest_classifier.predict(X_test)
    forest_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
    print(f"Accuracy using {name_method}:", forest_accuracy)

    std = tools_.get_std(clf)
    print("std_test_score best hyper parameters:", std)
    
    

# =============================================================================
#  STACKED
# =============================================================================

stacked_grid = load('fitted_model/stacked_grid.joblib')


y_pred = stacked_grid.predict(X_test)

print("++++++   STACKED CV BEST MODEL    ++++++")
stacked_grid_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using the stacked classifier:", stacked_grid_accuracy)

std = tools_.get_std(stacked_grid)
print("std_test_score best hyper parameters:", std)



print("\n")



# =============================================================================
#  FETAURES IMPORTANCE
# =============================================================================
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
grid_std = [None]*len(method_list)


print("++++++   FEATURE REDUCTION EXTRA TREES CV BEST MODELS    ++++++")
for i,method in enumerate(method_list):
    grid_cv[i]= load("fitted_model/dim_red_"+method["name_method"]+".joblib")
    grid_cv[i].best_estimator_.steps[-1][1].verbose = 0
    y_pred = grid_cv[i].predict(X_test)
    grid_accuracy[i] = accuracy_score(y_true = y_test, y_pred = y_pred)

    grid_std[i] = tools_.get_std(grid_cv[i])
           
for i,method in enumerate(method_list):    
    name_method = method["name_method"]
    print(f"Best accuracy using {name_method}: {grid_accuracy[i]}")
    best_reduce_dim = grid_cv[i].best_params_['reduce_dim__'+method["name_dim_param"]]
    # print(f"Best reduced number of features: {best_reduce_dim}")
    # grid_cv[i].best_params_['reduce_dim__'+method["name_dim_param"]]
    
    print("std_test_score best hyper parameters:", grid_std[i])


# =============================================================================
#  STACKED REDUCED NUMBER OF FEATURES
# =============================================================================

reduced_stacked_grid = load('fitted_model/stacked_red_feat_grid.joblib')
y_pred = reduced_stacked_grid.predict(X_test)


print("++++++   FEATURE REDUCTION STACKED CV BEST MODEL    ++++++")
reduced_stacked_grid_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using reducing features stacked classifier:", 
      reduced_stacked_grid_accuracy)

std = tools_.get_std(reduced_stacked_grid)
print("std_test_score best hyper parameters:", std)



# =============================================================================
#  COMPARING THE SIZE OF THE FILES
# =============================================================================
    

print("\n")
file = 'fitted_model/RandomForest_grid.joblib' 
size = tools_.get_file_size(file)
print("++++   RandomForest best model   ++++")     
tools_.convert_bytes(size, "MB")
file = 'fitted_model/ExtraTrees_grid.joblib'
size = tools_.get_file_size(file)
print("++++   ExtraTrees best model   ++++")     
tools_.convert_bytes(size, "MB")


print("\n")
file = 'fitted_model/knn_grid.joblib' 
size = tools_.get_file_size(file)
print("++++   KNN best model   ++++")     
tools_.convert_bytes(size, "MB")

file = 'fitted_model/pca_knn_rs.joblib' 
size = tools_.get_file_size(file)
print("++++   KNN with PCA feature selection best model   ++++")     
tools_.convert_bytes(size, "MB")


print("\n")
file = 'fitted_model/lin_svc_grid_l1.joblib' 
size = tools_.get_file_size(file)
print("++++   SVC penalty l1 best model   ++++")     
tools_.convert_bytes(size, "MB")

file = 'fitted_model/lin_svc_grid_l2.joblib' 
size = tools_.get_file_size(file)
print("++++   SVC penalty l2 best model   ++++")     
tools_.convert_bytes(size, "MB")


print("\n")
file = 'fitted_model/lr_classifier_grid_l1l2.joblib'  
size = tools_.get_file_size(file)
print("++++   LogisticRegression penalty l1 best model   ++++")     
tools_.convert_bytes(size, "MB")

file = 'fitted_model/lr_classifier_grid_el.joblib' 
size = tools_.get_file_size(file)
print("++++   LogisticRegression penalty elasticnet best model   ++++")     
tools_.convert_bytes(size, "MB")


print("\n")
file = 'fitted_model/stacked_grid.joblib'  
size = tools_.get_file_size(file)
print("++++   StackingClassifier best model   ++++")     
tools_.convert_bytes(size, "MB")



print("\n")
file = 'fitted_model/dim_red_SVC.joblib'  
size = tools_.get_file_size(file)
print("++++   ExtraTrees with SVC feature reduction   ++++")     
tools_.convert_bytes(size, "MB")

file = file = 'fitted_model/dim_red_random tree.joblib' 
size = tools_.get_file_size(file)
print("++++   ExtraTrees with RandomForestClassifier feature reduction   ++++")     
tools_.convert_bytes(size, "MB")

file = 'fitted_model/dim_red_PCA.joblib'  
size = tools_.get_file_size(file)
print("++++   ExtraTrees with PCA feature reduction   ++++")     
tools_.convert_bytes(size, "MB")


file = file = 'fitted_model/stacked_red_feat_grid.joblib' 
size = tools_.get_file_size(file)
print('\n')
print("++++   StackingClassifier with SVC feature reduction   ++++")     
tools_.convert_bytes(size, "MB")



# reason why the reduced feature stacking classifier has a bigger dimension that
# the stacking classifier:
print('\n\n')

n_nodes_et = 0
et_stack = stacked_grid.best_estimator_.named_estimators_['et']
for decision_tree in et_stack.steps[1][1].estimators_:
    n_nodes_et = n_nodes_et + decision_tree.tree_.node_count

print('Stacking classifier number of nodes ExtraTrees: ')
print(n_nodes_et)


n_nodes_red_feat_et = 0
stack_red_feat = reduced_stacked_grid.best_estimator_.named_steps['classify']
for decision_tree in stack_red_feat.estimators_[2].estimators_:
    n_nodes_red_feat_et = n_nodes_red_feat_et + decision_tree.tree_.node_count

print('Reduced feature stacking classifier number of nodes ExtraTrees: ')
print(n_nodes_red_feat_et)
