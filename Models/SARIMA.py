#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[2]:


df_train = pd.read_csv('../Dataset/Train_SU63ISt.csv')


# In[3]:


df_train['Datetime'] = pd.to_datetime(df_train['Datetime'])
df_train = df_train.sort_values(by=['Datetime'])


# In[5]:


sm.graphics.tsa.plot_acf(df_train['Count'].diff().dropna().values.squeeze(), lags=160)
plt.show()


# In[6]:


# from  the graph every 24 is higher correlation so S = 24


# In[7]:


SARIMA_model = sm.tsa.statespace.SARIMAX(df_train['Count'], order=(3, 1, 4),seasonal_order=(1, 1, 1, 24)).fit()
SARIMA_model.summary()


# In[8]:


# test 
df_test = pd.read_csv('../Dataset/Test_0qrQsBZ.csv')


# In[9]:


df_pred = SARIMA_model.predict(start=18288, end=23399)


# In[10]:


df_pred = df_pred.reset_index()
df_pred.rename(columns={0: 'Count', 'index':'ID'},
          inplace=True, errors='raise')


# In[11]:


final = pd.merge(df_test,df_pred,on='ID')


# In[12]:


final[['ID','Count']].to_csv('SARIMA.csv',index=False)


# In[13]:


# score = 306


# In[ ]:




