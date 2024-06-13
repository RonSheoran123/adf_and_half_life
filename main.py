import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

df=pd.read_csv('USDCAD_1day_sample.csv')

def adf_test(series):
    result=adfuller(series)
    print(f'ADF Statistic {result[0]}')
    print(f'p-value : {result[1]}')
    print(result[4])

adf_test(df['close'])  #p-value : 0.20232226802306458, can reject hypothesis in ~80% confidence interval

# half life
from sklearn import linear_model
df_lag = df['close'].shift(1)
df_delta = df['close'] - df_lag
lin_reg_model = linear_model.LinearRegression()
df_delta = df_delta.values.reshape(len(df_delta),1)                    # sklearn needs (row, 1) instead of (row,)
df_lag = df_lag.values.reshape(len(df_lag),1)
lin_reg_model.fit(df_lag[1:], df_delta[1:])                           # skip first line nan
half_life = -np.log(2) / lin_reg_model.coef_.item()
print ('Half life:       %s' % half_life)           #half life =  13.265894650483274 days

def SMA(data, period=30):
    data['sma']=data['close'].rolling(window=period).mean()

SMA(df,13)
df.dropna()

fig, ax=plt.subplots(figsize=(8,4))
ax.plot(df['close'])
ax.plot(df['sma'])
