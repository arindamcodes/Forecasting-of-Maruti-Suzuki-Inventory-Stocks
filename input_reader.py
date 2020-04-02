import pandas as pd
import datetime

def get_timeseries_data(APP,filename, con=None):
    print("filename :", filename)
    if APP==0:
        time_series_df = pd.read_csv("../Inputs/"+filename+".csv")
    else:
        time_series_df = con.read_table(filename)
        # time_series_df = xpd.read_sql_table(table_name=filename, con=engine, schema=schema)
        # print("Invalid Choice")
        # exit()
    time_series_df["time_period"] = time_series_df.apply(
        lambda x: datetime.datetime.strptime(x["time_period"], "%m-%d-%Y").date(), axis=1)
    return  time_series_df

def get_foreasting_methods(APP,filename, con=None):
    if APP==0:
        config_df = pd.read_csv("../Inputs/"+filename+".csv")
    else:
        config_df = con.read_table(filename)
        # config_df = pd.read_sql_table(table_name=filename, con=engine, schema=schema)
        # print("Invalid Choice")
        # exit()
    return  config_df


