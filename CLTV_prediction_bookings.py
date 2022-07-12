import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df_ = pd.read_csv("/Users/sedeftaskin/Desktop/Data_Science/archive/bookings.csv")
df = df_.copy()

df=df[(df["Adults"]!=0) | (df["Children"]!=0)]
df = df[df["RoomPrice"]>0]

# Missing Values
col_without_NaN = []
col_with_NaN = []
for col in df.columns:
    if any(df[col].isnull()):
        col_with_NaN.append(col)
    else:
        col_without_NaN.append(col)

# Filling them

df["Channel"].fillna("other", inplace=True)
df["Country"].fillna("otr", inplace=True)
# filling roomno median?
df.dropna(inplace=True)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    #dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df,"RoomPrice")
replace_with_thresholds((df,"TotalPayment")

for col in df.columns[df.columns.str.contains("Date")]:
    df[col] = pd.to_datetime(df[col])

df["DepartureDate"].max()
analysis_date = dt.datetime(2020, 9, 28)

cltv_df=df.groupby("GuestID").agg({"ArrivalDate": lambda x: (analysis_date - x.min()).days,
                                 "TotalPayment": "sum"})
cltv_df["frequency"]=df.groupby("GuestID").agg({"ArrivalDate":"nunique"})
cltv_df["recency"]=(df.groupby("GuestID")["DepartureDate"].max()-df.groupby("GuestID")["ArrivalDate"].min()).dt.days
cltv_df.head()

cltv_df.columns=["T","Monetary","frequency","recency"]
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"]=cltv_df["T"]/7
cltv_df["TotalPayment"]=cltv_df["TotalPayment"]/cltv_df['frequency']
cltv_df.head()
cltv_df.columns=["T-weekly","Monetary","frequency","recency_weekly"]
cltv_df.reset_index(inplace=True)
# BG/NBD for frequency
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_weekly'],
        cltv_df['T-weekly'])

cltv_df["exp_sale_3_months"] = bgf.predict(4 * 12,
                                               cltv_df['frequency'],
                                               cltv_df['recency_weekly'],
                                               cltv_df['T-weekly'])

cltv_df.sort_values("exp_sale_3_months",ascending=False)

# Gamma Gamma
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df[ 'Monetary'])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'], cltv_df['Monetary'])
cltv_df.head()

# CLTV

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_weekly'],
                                   cltv_df['T-weekly'],
                                   cltv_df['Monetary'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)
cltv.head()
cltv_final=pd.concat([cltv_df,cltv],axis=1)
cltv_final.sort_values(by="clv", ascending=False).head(20)
cltv_final.head(10)

# Segmentation

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()
