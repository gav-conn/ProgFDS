import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm

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

    lamb = float(np.divide((K+2)*(T-1)/(T-K-2), (mu-mu_mv * ones).T 
                     @ np.linalg.inv(sigma) @ (mu-mu_mv * ones)))
    gamma = lamb/(T+lamb)

    mu_bs = gamma * ones * mu_mv + (1-gamma) * mu
    sigma_bs = sigma * (1 + 1/(T+lamb)) + lamb/(
        T*(T+1+lamb)) * (ones @ ones.T /(ones.T @ np.linalg.inv(sigma) @ ones))
    return mu_bs, sigma_bs

def minvar_weights(returns):

    K = returns.shape[1]
    sigma = returns.cov()

    w_mv = np.divide(np.linalg.inv(sigma) @ np.ones((K,1)), np.ones((1,K)) @ np.linalg.inv(sigma) @ np.ones((K,1)))
    return w_mv

def tangency_weights(returns):

    K = returns.shape[1]
    mu = np.array(returns.mean()).reshape((K,1))
    sigma = returns.cov()

    w_tp = np.divide(np.linalg.inv(sigma) @ mu, np.ones((1,K)) @ np.linalg.inv(sigma) @ mu)
    return w_tp

def jobson_korkie_test(portfolio_returns, risk_free=0, acceptance_threshold=0.05):
        
    T = len(portfolio_returns)
    mu_a, mu_b = portfolio_returns.mean()-risk_free
    cov_ab = portfolio_returns.cov()

    sigma_a, sigma_b = np.diag(cov_ab)
    sigma_ab = cov_ab.iloc[1,0]

    theta = 1/T*(2*sigma_a*sigma_b-2*np.sqrt(sigma_a*sigma_b)*sigma_ab + 1/2*mu_a**2*sigma_b +
                  1/2*mu_b**2*sigma_a - mu_a*mu_b/(np.sqrt(sigma_a*sigma_b))*sigma_ab**2)
    c_jk = np.sqrt(sigma_b)*mu_a-np.sqrt(sigma_a)*mu_b

    jk_stat = c_jk/np.sqrt(theta)
    # Take negative of absolute value so we get consistent test results regardless of ordering of portfolio inputs
    jk_pval = norm.cdf(-abs(jk_stat))
    jk_reject_null = jk_pval < acceptance_threshold/2 # Reject if score in the 5% most extreme results (Two-way test) 
    return jk_stat, jk_pval, jk_reject_null

# Q1
df_hist = pd.DataFrame()
for ticker_name in tickers_list:

    hist = yf.download(ticker_name, start="2015-01-01", end="2022-12-31",  interval="1wk", progress=False)
    hist['Returns'] = hist['Adj Close'].pct_change()
    df_hist[ticker_name] = hist['Returns']
df_hist.drop(df_hist.index[0], inplace=True)

# Q2
print(f'Summary of ETF Returns:\n{returns_summary(df_hist)}')

# Q3
mu_bs, sigma_bs = bayes_stein(df_hist)
print(f'\nBayes-Stein Estimate of Mean Return:\n{mu_bs}')
print(f'\nBayes-Stein Estimate of Covariance of Return:\n{sigma_bs}')

# Q4
n = df_hist.shape[0]//2+1
df_train = df_hist.iloc[:n, :]
df_test = df_hist.iloc[n:, :]

w_mv = minvar_weights(df_train)
w_tp = tangency_weights(df_train)
print(f'\nMinimum variance portfolio weights\n{w_mv.T}')
print(f'\nTangency portfolio weights\n{w_tp.T}')

df_portfolio = pd.DataFrame()
df_portfolio['GMVP'] = df_test @ w_mv
df_portfolio['TP'] = df_test @ w_tp

print(df_portfolio.to_latex(float_format="%.4f"))

mu_portfolio = df_portfolio.mean()
sd_portfolio = df_portfolio.std()
sharpe_portfolio = round(mu_portfolio/sd_portfolio, 4)
print(f'\nSharpe Ratio of Portfolio:\n{sharpe_portfolio.to_string()}')

# Q5
jk_stat, jk_pval, jk_reject_null = jobson_korkie_test(df_portfolio)
print(f'\nTest Statistic: {jk_stat:.4f}')
print(f'Test P-Value: {jk_pval:.4f}')
print(f'Reject Null Hypothesis: {jk_reject_null}')
