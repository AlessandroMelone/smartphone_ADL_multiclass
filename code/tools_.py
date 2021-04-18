#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:32:53 2021

@author: alessandro
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
 
def get_std(classifier_cv):
    #find index of the best method in the cv_results array
    list_rank = classifier_cv.cv_results_['rank_test_score'].tolist()
    index_best = list_rank.index(1)
    
    return classifier_cv.cv_results_['std_test_score'][index_best]
    
    
    
def load_dataset():
    path = "../dataset/"
    train = pd.read_csv(os.path.join(path,'train.csv'))
    test = pd.read_csv(os.path.join(path,'test.csv'))
    
    # Removing '()' from column names
    columns = train.columns
    columns = columns.str.replace('[()]','')
    columns = columns.str.replace('[-]', '')
    columns = columns.str.replace('[,]','')
    
    train.columns = columns
    test.columns = columns
    
    # Taking from the dataset the lables and features
    y_train = train.Activity
    X_train = train.drop(['subject','Activity'], axis = 1)
    y_test = test.Activity
    X_test = test.drop(['subject','Activity'], axis = 1)
    return X_train, y_train, X_test, y_test;




# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import RandomizedSearchCV
plt.rcParams["font.family"] = 'DejaVu Sans'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def plot_confusion_matrices(name_method, name_parameter, y_test, y_pred):
    labels=['LAYING', 'SITTING','STANDING','WALKING','WALKING_DS','WALKING_US']
    fig = plt.figure(figsize=(3.2,3.2))
    plt.grid(b=False)
    plot_confusion_matrix(confusion_matrix(y_test.values, y_pred), 
                                  classes=labels,
                                  title=f"{name_method} {name_parameter} confusion matrix")
    fig.subplots_adjust(
        top=0.945,
        bottom=0.205,
        left=0.0,
        right=0.9,
        hspace=0.2,
        wspace=0.2
    )        
    plt.show()
    
    # normalized confusion matrix
    fig = plt.figure(figsize=(3.2,3.2))
    plt.grid(b=False)
    plot_confusion_matrix(confusion_matrix(y_test.values, y_pred), 
                                  classes=labels, normalize=True,
                                  title=f"{name_method} {name_parameter} confusion matrix (normalized)")
    fig.subplots_adjust(
        top=0.945,
        bottom=0.205,
        left=0.0,
        right=0.9,
        hspace=0.2,
        wspace=0.2
    )        
    plt.show()    







def plot_grid_search_2params(cv_results, grid_param_1, grid_param_2, 
                             name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
#     scores_mean
# array([0.88343392, 0.88833058, 0.87976174, 0.86752128])
    
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Accuracy Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

# Calling Method 
# plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 
# 'Max Features')


def plot_grid_search_1param(list_grid, grid_param_1, grid_param_2, name_param_1, 
                            name_param_2, labels = True):
    # Plot Grid search scores
    _, ax = plt.subplots(1,1)
    
    for idx , method in enumerate(list_grid):
        
        # Get Test Scores Mean and std for each grid search
        scores_mean = method.cv_results_['mean_test_score']    
    
        # Get Test Scores Mean and std for each grid search
        scores_mean = np.array(scores_mean)

        scores_sd = method.cv_results_['std_test_score']
        scores_sd = np.array(scores_sd)

        # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
        if labels:
            ax.plot(grid_param_1, scores_mean, '-o', label= name_param_2 
                    + ': ' + grid_param_2[idx])
        else:
            ax.plot(grid_param_1, scores_mean, '-o',)
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Accuracy Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    
    

    
def plot_grid_search(cv_grid, name_param_x, name_param_y, name_legend_x,
                     name_legend_y, x_log_scale=True):
    # Plot Grid search scores
    _, ax = plt.subplots(1,1)
    
    #list with all the parameters y
    # name_param_y  = 'lr__penalty' 
    array_y = cv_grid.param_grid[name_param_y]
    
    mean_array = cv_grid.cv_results_['mean_test_score']
    std_array = cv_grid.cv_results_['std_test_score']
    
    
    # y_value='l1'
    for y_value in array_y:
        grid_y_value = []
        grid_y_err = []
        # dict_param = {'lr__C': 0.0001, 'lr__penalty': 'l2'}
        for idx, dict_param in enumerate(cv_grid.cv_results_['params']):
            if (dict_param[name_param_y] == y_value):
                grid_y_value.append(mean_array[idx])         
                grid_y_err.append(std_array[idx])

        # ax.errorbar(cv_grid.param_grid[name_param_x],
        #              grid_y_value, yerr=grid_y_err, 
        #              label= f"{name_legend_y}: {y_value}")
        

        ax.plot(cv_grid.param_grid[name_param_x], 
                grid_y_value, '-o', label= f"{name_legend_y}: {y_value}")
        
    if x_log_scale:
        ax.set_xscale('log')

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_legend_x, fontsize=16)
    ax.set_ylabel('CV Accuracy Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')




def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


 
def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    elif unit == "GB":
        return print('File size: ' + str(round(size / (1024 * 1024 * 1024), 3)) + ' Gigabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')
 

