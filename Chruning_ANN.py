# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 23:20:03 2018

@author: sanket_padte
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import Imputer

Churning_train="train/churn_train.csv"
Churning_test="test/churn_train.csv"
Churning_pred="Validation/FinalPred.csv"
def load_churning_data(datapath):
    return pd.read_csv(datapath)

#load the dataset

Churning_train_dataset=load_churning_data(Churning_train)
Churning_train_dataset.head(100)
Churning_train_dataset.info()
#finding the data distribution in various states
Churning_train_dataset['st'].value_counts()
Churning_train_dataset.describe()
#find the variable distribution
Churning_train_dataset.hist(bins=50,figsize=(20,15))
Churning_train_dataset.nummailmes=Churning_train_dataset.nummailmes.replace(0,np.NaN)
Churning_train_dataset.hist(bins=50,figsize=(20,15))
#finding correlation matrix
corr_matrix=Churning_train_dataset.corr()
#creating imputer object
Churning_train_dataset.nummailmes.fillna(0)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(Churning_train_dataset.iloc[:, 5].reshape(-1,1))
Churning_train_dataset.iloc[:,5]=imputer.transform(Churning_train_dataset.iloc[: ,5].reshape(-1,1)).reshape(-1)
