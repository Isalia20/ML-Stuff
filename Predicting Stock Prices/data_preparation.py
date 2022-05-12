import pandas as pd
import yfinance as yf

# These stock tickers are retrieved for later usage. Currently predicting for MSFT only.
stock_tickers = ["MSFT", "GOOG", "AMZN", "TSLA",
                 "NVDA", "FB", "BABA", "INTC", "CRM", "ADM",
                 "PYPL", "ATVI", "EA", "TTD", "MTCH", "ZG", "YELP"]

price_history = pd.DataFrame(yf.Ticker("AAPL").history(period='2y', interval='1d', actions=False))
price_history.reset_index(inplace=True)
price_history["Ticker"] = "AAPL"

for ticker in stock_tickers:
    price_history_tick = yf.Ticker(ticker).history(period='2y',  # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
                                   interval='1d',  # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                                   actions=False)
    price_history_tick = pd.DataFrame(price_history_tick)
    price_history_tick.reset_index(inplace=True)
    price_history_tick["Ticker"] = ticker
    price_history = pd.concat([price_history, price_history_tick])

# Feature engineering starts here.
# We are going to predict the open price given the date and some features from the past.
MSFT_price_history = price_history[price_history["Ticker"] == "MSFT"]

MSFT_price_history["day"] = MSFT_price_history["Date"].dt.day
MSFT_price_history["month"] = MSFT_price_history["Date"].dt.month
MSFT_price_history["year"] = MSFT_price_history["Date"].dt.year
MSFT_price_history["weekday"] = MSFT_price_history["Date"].dt.weekday
MSFT_price_history["quarter"] = MSFT_price_history["Date"].dt.quarter
MSFT_price_history["day_of_year"] = MSFT_price_history["Date"].dt.dayofyear
MSFT_price_history["is_month_start"] = MSFT_price_history["Date"].dt.is_month_start
MSFT_price_history["is_month_end"] = MSFT_price_history["Date"].dt.is_month_end

MSFT_price_history.sort_values(by="Date", ascending=True, inplace=True)

lag_cols = ["Open", "Close", "Volume", "High", "Low"]

for lag_num in range(1,6):
    for col in lag_cols:
        MSFT_price_history["LAG_" + str(col) + "_" + str(lag_num)] = MSFT_price_history[col].shift(lag_num)

# Delete columns which won't be accessible in prediction. Close will be dropped later
cols_to_delete = ["High", "Low", "Volume"]

MSFT_price_history.drop(cols_to_delete, axis=1, inplace=True)




#Drop the last row (run this if market is open)
MSFT_price_history.drop(MSFT_price_history.index.max(), inplace=True)


# Make target binary 1 for increase 0 for decrease (for same 0)
MSFT_price_history["price_increase_bin"] = (MSFT_price_history["Open"] - MSFT_price_history["LAG_Close_1"]) > 0
MSFT_price_history["price_increase_bin"] = MSFT_price_history["price_increase_bin"].astype(int)

MSFT_price_history.drop(["Date", "Open", "Close", "Ticker"], axis=1, inplace=True)

MSFT_price_history.to_csv("MSFT_price_history.csv")