#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


# In[2]:


class dataTransformer:
    '''Transformer class to transform the data'''
    def __init__(self, col_name):
        self.featue_col = col_name
    
    def deriveDayOfTheMonth(self, X):
        X['day_of_month'] = X[self.featue_col].dt.day
        return X
    
    def deriveDayOfTheYear(self, X):
        X['day_of_year'] = X[self.featue_col].dt.dayofyear
        return X
    
    def deriveDayOfWeek(self, X):
        X['day_of_week'] = X[self.featue_col].dt.dayofweek
        return X
    
    def deriveQuarter(self, X):
        X['quarter'] = X[self.featue_col].dt.quarter
        return X
    
    def deriveMonth(self, X):
        X['month'] = X[self.featue_col].dt.month    
        return X
    
    def deriveHour(self, X):
        X['hour'] = X[self.featue_col].dt.hour
        return X
    
    def deriveYear(self, X):
        X['year'] = X[self.featue_col].dt.year
        return X
    
    def deriveWeekOfYear(self, X):
        X['weekofyear'] = X[self.featue_col].dt.weekofyear
        return X
    
    def deriveIsWeekeend(self, X):
        X["Is_Weekend"] = X.Datetime.dt.day_name().isin(['Saturday', 'Sunday']).astype(int)
        return X
    
    def transform(self, X, train = True):        
        X = self.deriveDayOfTheMonth(X)
        X = self.deriveDayOfTheYear(X)
        X = self.deriveDayOfWeek(X)
        X = self.deriveQuarter(X)
        X = self.deriveMonth(X)
        X = self.deriveHour(X)
        X = self.deriveYear(X)
        X = self.deriveWeekOfYear(X)
        X = self.deriveIsWeekeend(X)
        feat = X[['day_of_month','day_of_year','day_of_week','quarter','month',
                      'hour','year','weekofyear','Is_Weekend']]
        if train:
            label = X[['Count']]
            return feat, label
        else:
            return feat


# In[3]:


class model_def:
    def train_test_split(self, X, y, size = 0.25):
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=size)
        self.train_X = train_X
        self.test_X = test_X
        self.train_y = train_y
        self.test_y = test_y
        return
    
    def get_model(self, n_estimators=1000, learning_rate=0.05):
        self.model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
        return
    
    def fit_model(self, verbose=False):
        self.model.fit(self.train_X, self.train_y, 
             eval_set=[(self.test_X, self.test_y)], verbose=False)
        return
    
    def predict_future(self, test_df):
        self.predictions = self.model.predict(test_df)
        return


# In[10]:


df_train = pd.read_csv('../Dataset/Train_SU63ISt.csv')
df_train['Datetime'] = pd.to_datetime(df_train['Datetime'])
transformer = dataTransformer(col_name='Datetime')
X, y = transformer.transform(df_train, train = True)
model = model_def()
model.train_test_split(X, y)
model.get_model()
model.fit_model()
df_test = pd.read_csv('../Dataset/Test_0qrQsBZ.csv')
df_test['Datetime'] = pd.to_datetime(df_test['Datetime'])
X_test = transformer.transform(df_test, train = False)
model.predict_future(X_test)
y = pd.DataFrame(model.predictions, columns = ['Count'])
final = pd.concat([df_test['ID'],y],axis=1)
final.to_csv('XG.csv',index=False)


# In[11]:


final.shape


# In[12]:


df_test.shape


# In[ ]:




