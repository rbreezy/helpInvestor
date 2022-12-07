#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[ ]:


df_train = pd.read_csv('../Dataset/Train_SU63ISt.csv')


# In[ ]:


df_train['Datetime'] = pd.to_datetime(df_train['Datetime'])
df_train = df_train.sort_values(by=['Datetime'])


# In[ ]:


# Determining the P-value
# we can do that using PACF
sm.graphics.tsa.plot_pacf(df_train['Count'].diff().dropna().values.squeeze(), lags=20, method="ywm")
plt.show()


# In[ ]:


# we can see first time it is coming under blue region is when p=1


# In[ ]:


# Determining the Q-value
# we can do that using ACF
sm.graphics.tsa.plot_acf(df_train['Count'].diff().dropna().values.squeeze(), lags=20)
plt.show()


# In[ ]:


# we can see first time it is coming near to zero is when q=1


# In[ ]:


# D is 1 because it is only one diff


# In[50]:


from statsmodels.tsa.arima.model import ARIMA
ARIMA_model = ARIMA(df_train['Count'], order=(3, 2, 4)).fit()
ARIMA_model.summary()


# In[62]:


# test 
df_test = pd.read_csv('../Dataset/Test_0qrQsBZ.csv')


# In[63]:


df_pred = ARIMA_model.predict(start=18288, end=23399)


# In[64]:


df_pred = df_pred.reset_index()
df_pred.rename(columns={0: 'Count', 'index':'ID'},
          inplace=True, errors='raise')


# In[65]:


final = pd.merge(df_test,df_pred,on='ID')


# In[69]:


final[['ID','Count']].to_csv('ARIMA.csv',Index=False)


# In[15]:


# score = 340


# In[ ]:




