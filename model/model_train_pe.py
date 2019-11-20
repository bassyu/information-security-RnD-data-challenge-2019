#!/usr/bin/env python
# coding: utf-8

# In[1]:

def score(confusion_matrix):
    TP = float(confusion_matrix[0][0])
    FP = float(confusion_matrix[0][1])
    FN = float(confusion_matrix[1][0])
    TN = float(confusion_matrix[1][1])

    return (TP + TN) / sum(sum(confusion_matrix)) - ((FP / (FP + TN) * 0.6 + FN / (FN + TP) * 0.4))

def accuracy(confusion_matrix):
    return (confusion_matrix[0][0] + confusion_matrix[1][1]) / float(sum(sum(confusion_matrix)))


# In[2]:

import pandas as pd

data = pd.read_csv('../data/csv/pe_train.csv')

x_data = data.drop(['filename', 'class'], axis = 1)
y_data = data['class']

x_data.shape


# In[3]:

# tran:test = 8:2

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
                                                    test_size=0.2,
                                                    random_state=42)
x_train.shape[0], x_test.shape[0]


# In[4]:

# xgb
import xgboost as xgb

xgb = xgb.XGBClassifier(max_depth=10, 
                          learning_rate=0.05, 
                          objective= 'binary:logistic')

# random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()


# In[5]:

from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.externals import joblib 


model_name = {xgb: 'XGB', rf: 'RandomForest'}

model = xgb
model.n_estimators = 600

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
confusion_matrix = confusion_matrix(y_test, y_predict, labels=[1, 0])
print('model :', model, 
      'error :', mean_squared_error(y_test, y_predict), 
      'accuracy :', accuracy(confusion_matrix),
      'score :', score(confusion_matrix))

joblib.dump(model, '../data/pkl/model' + model_name[model] + '.pkl')
