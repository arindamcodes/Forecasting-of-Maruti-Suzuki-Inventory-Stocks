# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:03:08 2019
@author: Admin
"""
from pprint import pprint
import numpy as np

def get_timeseries_df(timeseries_data,sku,location):
    return timeseries_data[(timeseries_data["sku_id"]==sku) & (timeseries_data["location_id"]==location)]

def get_train_test_data(df,test_periods):
        filtered_sku_table = df[["time_period","demand"]].set_index(["time_period"],drop=True)
        # filtered_sku_table.index.freq = 'MS'
        train_data=filtered_sku_table[:-test_periods]
        test_data=filtered_sku_table[-test_periods:]
        return np.array(train_data).reshape(-1,1),np.array(test_data).reshape(-1,1)
def get_forecasting_data(df):
    filtered_sku_table = df[["time_period", "demand"]].set_index(["time_period"], drop=True)
    total_forecast_data=np.array(filtered_sku_table).reshape(-1,1)
    return total_forecast_data


def prepare_data(timeseries_data,config_data,test_periods):
    config_dict = config_data.set_index(["sku_id","location_id"]).to_dict()['forecasting_method']
    config_dict2 = config_data.set_index(["sku_id","location_id"]).to_dict()['seasonality_cycle']
    location_sku_pairs = list(config_dict.keys())
    forecasting_data={}
    for sku,location in location_sku_pairs:
        # forecasting_data[(location,sku)]
        df = get_timeseries_df(timeseries_data, sku,location)
        train,test=get_train_test_data(df,test_periods)
        previous_forecast_data=get_forecasting_data(df)
        forecasting_data[(sku,location)] = {"train":train,"test":test,"previous_forecast_data":previous_forecast_data, "method": config_dict[(sku,location)],
                                            "df":df, "season_cycle":config_dict2[(sku,location)]}
    return location_sku_pairs, forecasting_data

# import pandas as pd
# import numpy as np
# class InputFile(object):
#
#     def __init__(self,inputdata,SKU):
#         self.data=inputdata
#         self.sku=SKU
#
#     def get_sku_of_type(self):
#         distinct_sku = self.data["SKU"].unique()
#         self.data_dict = {}
#         for sku in distinct_sku:
#             self.data_dict[sku] = self.data[self.data["SKU"]==sku][['SKU','MMDDYYYY','SumOfNET_QTY']].reset_index(drop=True)
#         return distinct_sku, self.data_dict
#
#     def get_timeseries_data(self):
#         df=InputFile(self.data,self.sku)
#         df=df.get_sku_of_type()
#         sku_table=df[1][self.sku]
#         filtered_sku_table = sku_table[["MMDDYYYY","SumOfNET_QTY"]].set_index(["MMDDYYYY"],drop=True)
#         filtered_sku_table.index.freq = 'MS'
#         data = filtered_sku_table
#         return np.array(data).reshape(-1,1)
#
#     def get_train_test(self):
#         df=InputFile(self.data,self.sku)
#         df=df.get_sku_of_type()
#         sku_table=df[1][self.sku]
#         filtered_sku_table = sku_table[["MMDDYYYY","SumOfNET_QTY"]].set_index(["MMDDYYYY"],drop=True)
#         filtered_sku_table.index.freq = 'MS'
#         train = filtered_sku_table[:-6]
#         test = filtered_sku_table[-6:]
#         return np.array(train).reshape(-1,1),np.array(test).reshape(-1,1)

