import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt
%pylab inline
a=pd.read_excel("C:/Users/user/Desktop/INTERNSHIP/SOLAR_DADRI/DATA/COMBINED/06.xlsx",Header='None')

a.set_index=pd.to_datetime(a['Date'])
a.columns
Date=list(a['Date'])
Values=list(a['Values'])
b=pd.DataFrame({'Date':Date,'Values':Values})
b['Date']=pd.to_datetime(b['Date'])
b.set_index(b['Date'],inplace='True')
del b
ts=pd.Series(Values,index=pd.to_datetime(Date))
a=ts['2017-06']
b=a['2017-06-15 00:00:00 ':]


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
    
ts_log = np.log(ts)


moving_avg = ts_log.rolling(window=12,center=False).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

expwighted_avg = pd.ewma(ts_log, halflife=12)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)
