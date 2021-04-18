import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tools_
from joblib import dump
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

random_state = 42

X_train, y_train, X_test, y_test = tools_.load_dataset()
# =============================================================================
#  knn
# =============================================================================
estimators = [('scaler', StandardScaler()), 
              ('knn', KNeighborsClassifier())]

pipe = Pipeline(estimators)

n_neighbors = np.arange(2,40,1)
parameters = {'knn__n_neighbors' : n_neighbors}

knn_grid = GridSearchCV(pipe, param_grid=parameters, 
                      n_jobs=-1,verbose=1)

knn_grid.fit(X_train, y_train)

# knn_rs = load('fitted_model/knn_rs.joblib') 
y_pred = knn_grid.best_estimator_.predict(X_test)
knn_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using PCA and KNN", knn_accuracy)

dump(knn_grid, 'fitted_model/knn_grid.joblib') 

# =============================================================================
#  PCA dimension reduction + knn
# =============================================================================
estimators = [('scaler', StandardScaler()), 
              ('pca', PCA(random_state=random_state)),
              ('knn', KNeighborsClassifier())]

pipe = Pipeline(estimators)

#define RandomizedSearchCV
n_neighbors = np.arange(3,20,1)
n_components = np.arange(20,400,1)

parameters = {'pca__n_components' : n_components, 
              'knn__n_neighbors' : n_neighbors}

pca_knn_rs = RandomizedSearchCV(pipe, param_distributions = parameters,njobs=-1,
                                    random_state=42, verbose=1,n_iter=40)

pca_knn_rs.fit(X_train, y_train)

# pca_knn_rs = load('pca_knn_rs.joblib') 
y_pred = pca_knn_rs.best_estimator_.predict(X_test)
pca_knn_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using PCA and KNN", pca_knn_accuracy)

dump(pca_knn_rs, 'fitted_model/pca_knn_rs.joblib') 


# =============================================================================
# NeighborhoodComponentsAnalysis dimension reduction + knn (too much time to fit)
# =============================================================================
estimators = [('scaler', StandardScaler()), 
              ('NeighAnalysis', NeighborhoodComponentsAnalysis(random_state=random_state)),
              ('knn', KNeighborsClassifier())]

pipe = Pipeline(estimators)

#define RandomizedSearchCV
n_neighbors = np.arange(3,20,1)
n_components = np.arange(20,400,1)

parameters = {'NeighAnalysis__n_components' : n_components, 
              'knn__n_neighbors' : n_neighbors}

nca_knn_rs = RandomizedSearchCV(pipe, param_distributions = parameters,njobs=-1,
                                    random_state=42, verbose=1,n_iter=20)

nca_knn_rs.fit(X_train, y_train)

# kernel_svm_rs = load('fitted_model/kernel_svm_rs.joblib') 
y_pred = nca_knn_rs.best_estimator_.predict(X_test)
nca_knn_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using NeighborhoodComponentsAnalysis and KNN:", nca_knn_accuracy)

# dump(nca_knn_rs, 'fitted_model/nca_knn_rs.joblib') 

# =============================================================================
# NeighborhoodComponentsAnalysis dimension reduction + knn (memory preserving) 
# (too much time to fit)
# =============================================================================
from tempfile import mkdtemp
from shutil import rmtree

nca = NeighborhoodComponentsAnalysis(n_components=266,
                                      random_state=random_state)
estimators = [('scaler', StandardScaler()), 
              ('NeighAnalysis', nca),
              ('knn', KNeighborsClassifier())]

# pipe = Pipeline(estimators)

#define RandomizedSearchCV
n_neighbors = np.arange(3,30,1)
n_components = np.arange(20,400,1)

# parameters = {'NeighAnalysis__n_components' : n_components, 'knn__n_neighbors' : n_neighbors}
parameters = {'knn__n_neighbors' : n_neighbors}

cachedir = mkdtemp()
pipe = Pipeline(estimators, memory=cachedir,verbose=1)

nca_knn_rs = RandomizedSearchCV(pipe, param_distributions = parameters,
                                    random_state=42, verbose=1,n_iter=20)
# scoring = None 
# If None, the estimator's score method is used.

nca_knn_rs.fit(X_train, y_train)

# kernel_svm_rs = load('fitted_model/kernel_svm_rs.joblib') 
y_pred = nca_knn_rs.best_estimator_.predict(X_test)
nca_knn_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using NeighborhoodComponentsAnalysis and KNN:", nca_knn_accuracy)

# dump(nca_knn_rs, 'fitted_model/nca_knn_rs.joblib') 
# nca_knn_rs = load('fitted_model/nca_knn_rs.joblib') 

# Clear the cache directory when you don't need it anymore
rmtree(cachedir)


# ####
# # trasform the data once and train the the estimator with the traformed data, i.e.
# # not use pipeline

# =============================================================================
# NeighborhoodComponentsAnalysis dimension reduction + knn (single fit with best 
# params of the pca)
# =============================================================================

nca = NeighborhoodComponentsAnalysis(n_components=266,
                                      random_state=random_state)

estimators = [('scaler', StandardScaler()), 
              ('NeighAnalysis', nca),
              ('knn', KNeighborsClassifier(n_neighbors=14))]

pipe = Pipeline(estimators,verbose=1)

#define RandomizedSearchCV
n_neighbors = np.arange(3,20,1)
n_components = np.arange(20,400,1)

parameters = {'NeighAnalysis__n_components' : n_components, 'knn__n_neighbors' : n_neighbors}


pipe.fit(X_train, y_train)

# kernel_svm_rs = load('fitted_model/kernel_svm_rs.joblib') 
y_pred = pipe.predict(X_test)
nca_knn_accuracy = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Accuracy using NeighborhoodComponentsAnalysis and KNN:", nca_knn_accuracy)

nca_knn = pipe
dump(nca_knn, 'fitted_model/nca_knn.joblib') 
