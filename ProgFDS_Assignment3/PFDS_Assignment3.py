import pandas as pd
import yfinance as yf
import numpy as np

tickers_list = ['IVV', 'IEUR', 'AIA', 'IEMG']

def returns_summary(returns, riskfree_rate=0):

    mean_return     = returns.mean()
    median_return   = returns.median()
    sd_retrurns     = returns.std()
    skew_returns    = returns.skew()
    kurt_returns    = returns.kurt()
    sharpe_returns  = (mean_return-riskfree_rate)/sd_retrurns

    summary = pd.concat([mean_return, median_return, sd_retrurns, skew_returns, kurt_returns, sharpe_returns], axis=1)
    summary.columns = ['Mean', 'Median', 'Std. Dev.', 'Skewness', 'Kurtosis', 'Sharpe Ratio']
    return summary

def bayes_stein(returns):

    T, K = returns.shape
    ones = np.ones((K,1))
    mu = np.array(returns.mean()).reshape((K,1))    
    sigma = returns.cov()

    w_mv = minvar_weights(returns)

    mu_mv = w_mv.T @ mu

    lamb = np.divide((K + 2)*(T-1)/(T-K-2), (mu-mu_mv * ones).T @ np.linalg.inv(sigma) @ (mu-mu_mv * ones))[0,0]
    gamma = lamb/(T+lamb)

    mu_bs = gamma * ones * mu_mv + (1-gamma) * mu
    sigma_bs = sigma * (1 + 1/(T+lamb)) + lamb/(T*(T+1+lamb)) * (ones @ ones.T /(ones.T @ np.linalg.inv(sigma) @ ones))
    return mu_bs, sigma_bs

def minvar_weights(returns):

    K = returns.shape[1]
    sigma = returns.cov()

    numer = np.linalg.inv(sigma) @ np.ones((K,1))
    denom = np.ones((1,K)) @ np.linalg.inv(sigma) @ np.ones((K,1))

    w_mv = np.divide(numer, denom)
    return w_mv

def tangency_weights(returns):

    K = returns.shape[1]
    mu = np.array(returns.mean()).reshape((K,1))
    sigma = returns.cov()

    numer = np.linalg.inv(sigma) @ mu
    denom = np.ones((1,K)) @ np.linalg.inv(sigma) @ mu

    w_mv = np.divide(numer, denom)
    return w_mv

df_hist = pd.DataFrame()

for ticker_name in tickers_list:

    yticker = yf.Ticker(ticker_name)
    hist = yticker.history(start="2015-01-01", end="2022-12-31",  interval="1wk")

    hist['Returns'] = hist['Close'].pct_change()
    df_hist[ticker_name] = hist['Returns']

df_hist.drop(df_hist.index[0], inplace=True)

print('Summary of ETF Returns:\n', returns_summary(df_hist))

bs_estimates = bayes_stein(df_hist)
print('\nBayes-Stein Estimate of Mean Return:\n', bs_estimates[0], 
      '\nBayes-Stein Estimate of Covariance of Return:\n', bs_estimates[1])

n = df_hist.shape[0]//2+1
df_train = df_hist.iloc[:n, :]
df_test = df_hist.iloc[n:, :]

w_mv = minvar_weights(df_train)
w_tp = tangency_weights(df_train)

df_portfolio = pd.DataFrame()
df_portfolio['GMVP'] = df_test @ w_mv
df_portfolio['TP'] = df_test @ w_tp

mu_portfolio = df_portfolio.mean()
sd_portfolio = df_portfolio.std()
sharpe_portfolio = mu_portfolio/sd_portfolio
print('\nSharpe Ratio of Portfolio:\n', sharpe_portfolio)
