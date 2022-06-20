#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import math


# In[2]:


x_train = pd.read_csv("./data/x_train.csv")
y_train = pd.read_csv("./data/y_train.csv")
x_test  = pd.read_csv("./data/x_test.csv")
y_test  = pd.read_csv("./data/y_test.csv")


# In[3]:


#concatenate dfs
train_df = pd.concat([x_train, y_train], axis=1)
test_df = pd.concat([x_test, y_test], axis=1)


# In[4]:


x_train[x_train > 0] = 1
x_test[x_test > 0] = 1


# In[5]:


#calculate MLE estimators
spam_word_sum = x_train[train_df["Prediction"] == 1].sum().sum()
theta_spam = x_train[train_df["Prediction"] == 1].sum().values / spam_word_sum

normal_word_sum = x_train[train_df["Prediction"] == 0].sum().sum()
theta_normal = x_train[train_df["Prediction"] == 0].sum().values / normal_word_sum

pi_normal = len(y_train[ y_train["Prediction"] == 0]) / len(y_train)


# In[6]:


def classify(document_features):
    feature_arr = document_features.values
    probs = {}
    
    log_spam = np.log(feature_arr*theta_spam + (1 - feature_arr)*(1 - theta_spam))
    log_spam[log_spam == -math.inf] = -1e12
    probs[1] = np.log(1 - pi_normal) + np.sum(log_spam)
    
    
    log_normal = np.log(feature_arr*theta_normal + (1 - feature_arr)*(1 - theta_normal))
    log_normal[ log_normal == -math.inf] = -1e12
    probs[0] = np.log(pi_normal) + np.sum(log_normal)
    
    return max(probs, key=probs.get)
    
    


# In[7]:


def performance_metrics(test_x, test_y):
    prediction = test_x.apply(classify, axis=1)
    
    confusion_matrix = pd.crosstab(test_y["Prediction"], prediction , rownames=['Actual'], colnames=['Predicted'])  
    
    tp = confusion_matrix.iloc[1][1]
    tn = confusion_matrix.iloc[0][0]
    fp = confusion_matrix.iloc[0][1]
    fn = confusion_matrix.iloc[1][0]
    
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp /(tp + fp)
    f_measure = 2*precision*recall/(precision + recall)

    results = { "TP":tp, "TN":tn, "FP":fp, "FN":fn, "Accuracy":accuracy, "Recall":recall, "Precision":precision, "F-measure": f_measure}
    return results
    
    


# In[8]:


result = performance_metrics(x_test, y_test)


# In[9]:


output =f"""
Bernoulli Naive Bayes Model:
TP: {result["TP"]} FP: {result["FP"]}
FN: {result["FN"]} TN: {result["TN"]}

Accuracy:   {result["Accuracy"]}
Recall:     {result["Recall"]}
Precision:  {result["Precision"]}
F-measure:  {result["F-measure"]}

Wrong predicions: {result["FP"] + result["FN"]}
"""

print(output)

