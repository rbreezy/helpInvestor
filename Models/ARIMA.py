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


# In[21]:


# Determining the P-value
# we can do that using PACF
sm.graphics.tsa.plot_pacf(df_train['Count'].diff().dropna().values.squeeze(), lags=20, method="ywm")
plt.show()


# In[22]:


# we can see first time it is coming under blue region is when p=1


# In[23]:


# Determining the Q-value
# we can do that using ACF
sm.graphics.tsa.plot_acf(df_train['Count'].diff().dropna().values.squeeze(), lags=20)
plt.show()


# In[24]:


# we can see first time it is coming near to zero is when q=1


# In[25]:


# D is 1 because it is only one diff


# In[ ]:




