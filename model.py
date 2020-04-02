# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:14:35 2019
@author: Arindham Sarkar (+91-7797703756)
"""
import numpy as np
import pandas as pd
import math
from pyramid.arima import auto_arima
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.model_selection import GridSearchCV
import keras
# import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Dropout
from sklearn import linear_model


# from keras.models import load_model
def fourier(lengthrange, frequency, timeperiod):
    # initialize a null dataframe
    leng = len(lengthrange)
    xreg1 = np.zeros((leng, frequency))
    xreg2 = np.zeros((leng, frequency))
    xreg = np.zeros((leng, 2 * frequency))
    for i in lengthrange:
        for j in range(1, frequency + 1):
            xreg1[i - lengthrange[0], j - 1] = math.sin(2 * math.pi * i * j / timeperiod)
            xreg2[i - lengthrange[0], j - 1] = math.cos(2 * math.pi * i * j / timeperiod)
    np.concatenate((xreg1, xreg2), axis=1, out=xreg)
    return xreg



class Lstm(object):

    def __init__(self,dataset,season_cycle):
        self.cycle=season_cycle
        self.data=dataset
        self.data2=dataset
        self.X=[] 
        self.min=self.data.min()
        self.max=self.data.max()

    def modelfit(self):
        sc = MinMaxScaler(feature_range=(0, 1))
        self.data=sc.fit_transform(self.data)
        y= []
        for i in range(4,len(self.data)):
            self.X.append(self.data[i-4:i, 0])
            y.append(self.data[i, 0])
        self.X,y = np.array(self.X), np.array(y)
        self.X= np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
        regressor = Sequential()
        regressor.add(LSTM(units = 30, return_sequences = True,activation='relu', input_shape = (self.X.shape[1], 1)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 30,return_sequences = True,activation='relu',kernel_initializer='glorot_uniform',recurrent_activation='hard_sigmoid'))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = 30,activation='relu',kernel_initializer='glorot_uniform',recurrent_activation='hard_sigmoid'))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1,activation='linear'))
        adam=keras.optimizers.Adam(lr=0.0001,epsilon=0.000000001, decay=0,amsgrad=True)
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error' )
        regressor.fit(self.X, y, epochs = 500,batch_size=20,verbose=0)
        return regressor

    def test_Forecast(self,nmonths):
        sc = MinMaxScaler(feature_range=(0, 1))
        regr = linear_model.LinearRegression()
        ix=range(0,len(self.data))
        ix=np.array(ix).reshape(-1,1)
        ix2=range(0,len(self.data)+nmonths)
        ix2=np.array(ix2).reshape(-1,1)
        trend_fit=regr.fit(ix,self.data)
        trend=trend_fit.predict(ix)
        trend=trend.reshape(-1,1)
        trend_forecast=trend_fit.predict(ix2)
        trend_forecast=trend_forecast.reshape(-1,1)
        self.data=self.data-trend
        m1=Lstm(self.data,self.cycle)
        m2=m1.modelfit()
        data1=self.data
        data1=sc.fit_transform(data1)
        l=len(data1)
        for j in range(l,l+nmonths):
            self.X=[]
            for i in range(j,j+1):
                 self.X.append(data1[i-4:i, 0])
            self.X = np.array(self.X)
            self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
            forecasted_scaled=m2.predict(self.X)
            data1=np.concatenate((data1,forecasted_scaled))
        forecasted_scaled=data1[l:,0].reshape(-1,1)
        forecasted=sc.inverse_transform(forecasted_scaled)
        self.forecast=forecasted+trend_forecast[l:l+nmonths]
        return self.forecast
    
    def Forecast(self,nmonths):
        sc = MinMaxScaler(feature_range=(0, 1))
        sc = MinMaxScaler(feature_range=(0, 1))
        regr = linear_model.LinearRegression()
        ix=range(0,len(self.data))
        ix=np.array(ix).reshape(-1,1)
        ix2=range(0,len(self.data)+nmonths)
        ix2=np.array(ix2).reshape(-1,1)
        trend_fit=regr.fit(ix,self.data)
        trend=trend_fit.predict(ix)
        trend=trend.reshape(-1,1)
        trend_forecast=trend_fit.predict(ix2)
        trend_forecast=trend_forecast.reshape(-1,1)
        self.data=self.data-trend
        m1=Lstm(self.data,self.cycle)
        m2=m1.modelfit()
        data1=self.data
        data1=sc.fit_transform(data1)
        l=len(data1)
        for j in range(4,l+nmonths):
            self.X=[]
            for i in range(j,j+1):
                 self.X.append(data1[i-4:i, 0])
            self.X = np.array(self.X)
            self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
            forecasted_scaled=m2.predict(self.X)
            data1=np.concatenate((data1,forecasted_scaled))
        forecasted_scaled=data1[l:,0].reshape(1,-1)
        forecasted=sc.inverse_transform(forecasted_scaled)
        final_forecast=np.array(forecasted).reshape(-1,1)
        final_forecast=np.concatenate((self.data[0:4],final_forecast))
        final_forecast=final_forecast+trend_forecast
        return final_forecast
        
    
    
    def Wmape(self,test):
           forecast=self.forecast
           e=(sum(np.abs(forecast - test))/sum(test))*100
           return e
       
class Gru(object):
    def __init__(self,dataset,season_cycle):
        self.data=dataset
        self.X=[] 
        self.data2=dataset
        self.min=self.data.min()
        self.max=self.data.max()
        self.cycle=season_cycle
    def modelfit(self):
        sc = MinMaxScaler(feature_range=(0, 1))
        self.data=sc.fit_transform(self.data)
        y= []
        for i in range(4,len(self.data)):
            self.X.append(self.data[i-4:i, 0])
            y.append(self.data[i, 0])
        self.X,y = np.array(self.X), np.array(y)
        self.X= np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
        regressor = Sequential()
        regressor.add(GRU(units = 30, return_sequences = True,activation='tanh', input_shape = (self.X.shape[1], 1)))
        regressor.add(Dropout(0.2))
        regressor.add(GRU(units = 15,return_sequences = True,activation='tanh',kernel_initializer='glorot_uniform',recurrent_activation='hard_sigmoid'))
        regressor.add(Dropout(0.2))
        regressor.add(GRU(units = 15,return_sequences = True,activation='tanh',kernel_initializer='glorot_uniform',recurrent_activation='hard_sigmoid'))
        regressor.add(Dropout(0.2))
        regressor.add(GRU(units = 30,activation='tanh',kernel_initializer='glorot_uniform',recurrent_activation='hard_sigmoid'))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1,activation='linear'))
        adam=keras.optimizers.Adam(lr=0.0001,epsilon=0.000000001, decay=0,amsgrad=True)
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.fit(self.X, y, epochs=500,batch_size=20,verbose=0)
        return regressor
    
    def test_Forecast(self,nmonths):
        self.test_periods = nmonths
        sc = MinMaxScaler(feature_range=(0, 1))
        m1=Gru(self.data,self.cycle)
        m2=m1.modelfit()
        data1=self.data
        data1=sc.fit_transform(data1)
        l=len(data1)
        for j in range(l,l+nmonths+1):
            self.X=[]
            for i in range(j,j+1):
                 self.X.append(data1[i-4:i, 0])
            self.X = np.array(self.X)
            self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
            forecasted_scaled=m2.predict(self.X)
            data1=np.concatenate((data1,forecasted_scaled))
        forecasted_scaled=data1[l+1:,0].reshape(1,-1)
        forecasted=self.min+(self.max-self.min)*forecasted_scaled
        self.forecast=np.array(forecasted).reshape(-1,1)
        return self.forecast
    
    def Forecast(self,nmonths):
        sc = MinMaxScaler(feature_range=(0, 1))
        m1=Gru(self.data,self.cycle)
        m2=m1.modelfit()
        data1=self.data
        data1=sc.fit_transform(data1)
        l=len(data1)
        for j in range(4,l+nmonths):
            self.X=[]
            for i in range(j,j+1):
                 self.X.append(data1[i-4:i, 0])
            self.X = np.array(self.X)
            self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1], 1))
            forecasted_scaled=m2.predict(self.X)
            data1=np.concatenate((data1,forecasted_scaled))
        forecasted_scaled=data1[l:,0].reshape(1,-1)
        forecasted=self.min+(self.max-self.min)*forecasted_scaled
        final_forecast=np.array(forecasted).reshape(-1,1)
        final_forecast=np.concatenate((self.data2[0:4],final_forecast))
        return final_forecast
    
    def Wmape(self,test):
           forecast=self.forecast
           e=(sum(np.abs(forecast - test))/sum(test))*100
           return e
    
       
class SArima(object):
    def __init__(self,dataset,season_cycle):
        self.data=dataset
        self.cycle=season_cycle
    def modelfit(self):
        regressor=auto_arima(self.data,start_p=1, start_q=1,
                   max_p=6, max_q=3, stationary=False,
                   seasonal=True, max_d=2, trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True, maxiter=5000)
        regressor.fit(self.data)
        return regressor
    def excution(self):
        m1=SArima(self.data,self.cycle)
        m2=m1.modelfit()
        return m2
    def test_Forecast(self,nmonths):
        self.test_periods = nmonths
        m1=SArima(self.data,self.cycle)
        m2=m1.excution()
        forecasted=m2.predict(nmonths)
        self.forecast = forecasted.reshape(-1,1)
        return
    def Forecast(self,nmonths):
        m1=SArima(self.data,self.cycle)
        m2=m1.excution()
        previous_forecasted=m2.arima_res_.fittedvalues
        previous_forecasted=np.array(previous_forecasted).reshape(-1,1)
        future_forecast= m2.predict(nmonths)
        future_forecast=future_forecast.reshape(-1,1)
        final_forecast=np.concatenate((previous_forecasted,future_forecast))
        return final_forecast
        
    
    def Wmape(self,test):
        forecast=self.forecast
        e=(sum(np.abs(forecast - test))/sum(test))*100
        return e
        
        
class exponentialsmoothing(object):
    def __init__(self,dataset,season_cycle):
        self.data=dataset
        self.cycle=season_cycle
    def modelfit(self):
        regressor = ExponentialSmoothing(self.data, trend='add', seasonal = 'add',damped=True, seasonal_periods = self.cycle).fit()
        return regressor
    def test_Forecast(self,nmonths):
        es=exponentialsmoothing(self.data,self.cycle)
        model=es.modelfit()
        self.forecast=np.array(model.forecast(nmonths)).reshape(nmonths,1)
        return self.forecast
    def Forecast(self,nmonths):
        es=exponentialsmoothing(self.data,self.cycle)
        model=es.modelfit()
        final_forecast=np.array(model.predict(start=0,end=len(self.data)+nmonths)).reshape(-1,1)
        return final_forecast
        
    def Wmape(self,test):
           forecast=self.forecast
           e=(sum(np.abs(forecast - test))/sum(test))*100
           return e  
        
        
        
        
    
        
    
            
        
        
        
      