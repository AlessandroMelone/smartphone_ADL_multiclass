#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:12:22 2021

@author: alessandro
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns # for visualization

# =============================================================================
# importing dataset
# =============================================================================
path = "../dataset/"
train = pd.read_csv(os.path.join(path,'train.csv'))
test = pd.read_csv(os.path.join(path,'test.csv'))

train.describe()
# train.shape
# test.shape


print('Number of duplicates in the training set: {}'.format(sum(train.duplicated())))
print('Number of duplicates in the test set: {}'.format(sum(test.duplicated())))

print('Number of NaN/Null in the training set: {}'.format(train.isnull().values.sum()))
print('Number of NaN/Null in the test set: {}'.format(test.isnull().values.sum()))

#check if the dataset is particularized for some user
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Dejavu Sans'
plt.figure(figsize=(16,8))
plt.title('Amount of data per user', fontsize=20)
sns.countplot(x='subject',hue='Activity', data = train)
plt.show()

#check if the observation for each 
plt.figure(figsize = (12,8))
activity = train['Activity'].groupby(train['Activity']).count().index
activity_data = train['Activity'].groupby(train['Activity']).count().values
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b","#a4d321"]
plt.pie(activity_data, labels=activity,  colors=colors , autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()





