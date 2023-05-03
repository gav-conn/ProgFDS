import yahooquery as yq
import pandas as pd
import numpy as np
from splinter import Browser
import time

# Set input variables
download_folder = r'C:\Users\Gavin\Downloads'
# Set path for browser driver
executable_path = {'executable_path': r'C:\Users\Gavin\Desktop\geckodriver-v0.33.0-win64\geckodriver.exe'}

# Link to FRED database
url = 'https://fred.stlouisfed.org/series/SP500'

start_date = '1979-12-01'
end_date = '2021-12-31'

num_trading_days = 252
ma_lag = 10


# Download S&P data from FRED website 
# Comment out lines between dashes to run file using pre-downloaded file
# ------------------------------------------------------------- 
# Download the FRED data for the given dates from website using the splinter package.
with Browser('firefox', **executable_path, headless=False) as browser: 
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