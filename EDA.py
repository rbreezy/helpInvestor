#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[2]:


df = pd.read_csv('Dataset/Train_SU63ISt.csv')


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values(by=['Datetime'])


# In[6]:


plt.title('visual of time series data')
plt.plot(df.Datetime, df.Count)


# In[7]:


from statsmodels.tsa.stattools import adfuller
dftest = adfuller(df.Count,autolag='AIC')


# In[17]:


round(dftest[1],4)


# In[9]:


# Null: Data is not stationary
# p-value is less than 0.05 so can not reject

