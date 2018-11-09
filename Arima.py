
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6


# In[2]:


dataset = pd.read_csv('C:/Users/sachin.ichake/Downloads/AirPassengers.csv',infer_datetime_format=True)
indexedDataset = dataset.set_index('Month')
indexedDataset.head(5)


# In[3]:


plt.xlabel('Date')
plt.ylabel('Number of air passengers')
plt.plot(indexedDataset)


# In[4]:


#Determine the rolling statistics
rolmean = indexedDataset.rolling(window=12).mean()
rolstd = indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)


# In[5]:


#Plot rolling stats
orig = plt.plot(indexedDataset,color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label ='Rolling mean')
std = plt.plot(rolstd,color = 'black', label ='Rolling std')
plt.legend(loc = 'best')
plt.title('Rolling mean and Standard Deviation')
plt.show()


# In[6]:


# perform dickey-fuller test
from statsmodels.tsa.stattools import adfuller
print ('Result for Dickey fuller test')
# print(indexedDataset['#Passengers'])
dftest= adfuller(indexedDataset['#Passengers'],autolag='AIC')
dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of observation Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key]=value
    
print(dfoutput)
    


# In[22]:


#Estimate trend
indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)


# In[8]:


#MA
movingAverage = indexedDataset_logScale.rolling(window=12).mean()
movingStd = indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage,color='red')


# In[9]:


datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)


# In[10]:


from statsmodels.tsa.stattools import adfuller
def test_stationary(timeseries):
    movingAverage = timeseries.rolling(window=12).mean()
    movingStd = timeseries.rolling(window=12).std()
    
    # Plot rolling
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(movingAverage, color='red',label='Rolling Average')
    std = plt.plot(movingStd, color='black',label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # perform Dickey Fuller Test
    print('Result of Dickey Fuller Test')
    dftest = adfuller(timeseries['#Passengers'],autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of observation Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key]=value    
    print(dfoutput)


# In[11]:


test_stationary(datasetLogScaleMinusMovingAverage)


# In[12]:


exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage,color='red')


# In[13]:


datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
test_stationary(datasetLogScaleMinusMovingExponentialDecayAverage)


# In[14]:


datasetLogDiffShifting  = indexedDataset_logScale-indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)


# In[15]:


datasetLogDiffShifting.dropna(inplace=True)
test_stationary(datasetLogDiffShifting)


# In[17]:


from statsmodels.tsa.seasonal import seasonal_decompose

# print(indexedDataset_logScale)
indexedDataset_logScale.reset_index(inplace=True)
indexedDataset_logScale['Month'] = pd.to_datetime(indexedDataset_logScale['Month'])
indexedDataset_logScale = indexedDataset_logScale.set_index('Month')
# s=sm.tsa.seasonal_decompose(df.divida)

decomposition = seasonal_decompose(indexedDataset_logScale)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Resudual')
plt.legend(loc='best')
plt.tight_layout()



# In[18]:


decomposedLogData =residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)


# In[19]:


from statsmodels.tsa.stattools import acf , pacf
lag_acf = acf(datasetLogDiffShifting,nlags=20)
lag_pacf = pacf(datasetLogDiffShifting,nlags=20,method='ols')

#plot acf
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')


#plot acf
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')


# In[26]:


from statsmodels.tsa.arima_model import ARIMA
#AR Model

model = ARIMA(indexedDataset_logScale,order=(2, 1, 2))

# print(model)
result_AR= model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(result_AR.fittedvalues,color='red')
plt.title('RSS: %.4f'%sum((result_AR.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
print('Plotting AR Model')


# In[27]:


from statsmodels.tsa.arima_model import ARIMA
#MA Model

model = ARIMA(indexedDataset_logScale,order=(2, 1, 0))

# print(model)
result_MA= model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(result_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'%sum((result_MA.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
print('Plotting AR Model')


# In[28]:


from statsmodels.tsa.arima_model import ARIMA
#MA Model

model = ARIMA(indexedDataset_logScale,order=(2, 1, 2))

# print(model)
result_ARIMA= model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(result_ARIMA.fittedvalues,color='red')
plt.title('RSS: %.4f'%sum((result_ARIMA.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
print('Plotting AR Model')


# In[29]:


prediction_ARIMA_diff = pd.Series(result_ARIMA.fittedvalues,copy=True)
print(prediction_ARIMA_diff.head())


# In[32]:


prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
print(prediction_ARIMA_diff_cumsum.head())


# In[34]:


prediction_ARIMA_log = pd.Series(indexedDataset_logScale['#Passengers'].ix[0], index=indexedDataset_logScale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum,fill_value =0)
prediction_ARIMA_log.head()


# In[35]:


prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(prediction_ARIMA)


# In[37]:


indexedDataset_logScale


# In[38]:


result_ARIMA.plot_predict(1,264)

