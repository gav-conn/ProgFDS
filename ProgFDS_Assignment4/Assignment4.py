import yahooquery as yq
import pandas as pd
import numpy as np
from splinter import Browser
import time
from scipy.stats import norm

# Set input variables
download_folder = r'Z:\Users\Gavin\Downloads'
# Set path for browser driver
executable_path = {'executable_path': r'C:\Users\Gavin\Desktop\geckodriver.exe'}

# Link to FRED database
url = 'https://fred.stlouisfed.org/series/DTB4WK'

start_date = '2018-04-24'
end_date = '2023-04-21'

num_trading_days = 252
ma_lag = 10


# Download SPY data using yahooquery
# (yfinance package has been lacking some functionality due to changes to the yahoofinance site)
spy = yq.Ticker("SPY")
df = spy.history(start=start_date, end=end_date, interval='1d')
df['returns'] = np.log(df['adjclose'].pct_change()+1)

df.reset_index(drop=False, inplace=True)
df = df[['date', 'adjclose', 'returns']]

df['ma'] = df['adjclose'].rolling(window = ma_lag).mean()

print(df.head(10))

# Comment out lines between dashes to run file using pre-downloaded file
# ------------------------------------------------------------- 
# Download the FRED data for the given dates from website using the splinter package.
with Browser('firefox', **executable_path, headless=True) as browser: 
    browser.visit(url)
    browser.find_by_id('input-cosd').click()
    time.sleep(1)
    browser.find_by_id('input-cosd').fill(start_date)
    browser.find_by_id('input-coed').click()
    time.sleep(1)
    browser.find_by_id('input-coed').fill(end_date)
    browser.find_by_id('download-button').click()
    browser.find_by_id('download-data-csv').click()
# ------------------------------------------------------------- 

df_rf = pd.read_csv(download_folder+"\\DTB4WK.csv")

# Transform the risk-free rate to daily rate, & fill in '.' values with previous value.
df_rf['DTB4WK'].loc[df_rf['DTB4WK']=='.'] = np.nan
df_rf['DTB4WK'] = df_rf['DTB4WK'].fillna(method='ffill')

df_rf['DTB4WK'] = df_rf['DTB4WK'].astype(float)/100
df_rf['DTB4WK'] = (1+df_rf['DTB4WK'])**(1/num_trading_days)-1

df_rf.rename(columns={"DATE":"date","DTB4WK":"rf"}, inplace=True)
df_rf['date'] = pd.to_datetime(df_rf['date']).dt.date

# Join risk-free rate & SPY data
df = df.merge(df_rf, how='left', on='date')


# Calculate strategy position
df['position']=-1
df.loc[df['adjclose']>df['ma'], 'position']=1
df['position'] = df['position'].shift(1)

df['excess_return']=0
df.loc[df['position']==1, 'excess_return']=df['returns']-df['rf']

# Calculate summary statistics of returns
mean_return     = df['excess_return'].mean()
vol_returns     = df['excess_return'].std()
sharpe_returns  = mean_return/vol_returns

annualized_return = (1+mean_return)**num_trading_days-1
annualized_vol = vol_returns*np.sqrt(num_trading_days)
annualized_sharpe = annualized_return/annualized_vol

summary = pd.DataFrame(np.stack([[mean_return, vol_returns, sharpe_returns], 
                                 [annualized_return, annualized_vol, annualized_sharpe]]))

summary.index = ['Daily', 'Annual']
summary.columns = ['Mean', 'Volatility', 'Sharpe Ratio']
print(summary)


# Perform Pearsan-Timmermann test on the predicted sign of returns
df_pt_test = df.dropna()[['returns','position']] # Drop observations in the MA lead-in
n = len(df_pt_test)

df_pt_test['Z'] = np.sign(df_pt_test['returns']*df_pt_test['position'])
df_pt_test.loc[df_pt_test['Z']<0, 'Z']=0 # convert negative values to 0

P_hat = df_pt_test['Z'].mean()
P_x = len(df_pt_test.loc[df_pt_test['returns']>0])/n
P_y = len(df_pt_test.loc[df_pt_test['position']>0])/n
P_star = P_x*P_y+(1-P_x)*(1-P_y)

var_P_hat = n**-1*P_star*(1-P_star)
var_P_star = n**-1*(2*P_y-1)**2*P_x*(1-P_x) + n**-1*(2*P_x-1)**2*P_y*(1-P_y) + 4*n**-2*P_y*P_x*(1-P_y)*(1-P_x)

S_n = (P_hat-P_star)/(var_P_hat-var_P_star)**(1/2)
p_val = norm.cdf(-abs(S_n))

print(f'\nPearsan-Timmermann Test Statistic:\t{S_n:.4f}\nP-value of Test:\t{p_val:.4f}')