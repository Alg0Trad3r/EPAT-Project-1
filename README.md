# EPAT-Project-1

QuantInsti EPAT (Executive Programme in Algorithmic Trading) Project 1

Python Version Dependencies: Python 3.14.3 or 3.13

This is the first assigned project from QuantInsti's EPAT Program with the goal to develop a set of backtesting engines.

All of the code in this repository was written by me. No AI help other than research of specific use case syntax concepts that I then took and
applied to my project.

You will find the data I used for the project in a folder inside the project repository. You may need to change the path of the data being referenced
to to get the code actually running

Here is the direct prompt from QuantInsti with the requirements and goals of each individual strategy.

I. Backtests a strategy using three moving averages on any indices such as Nifty50, SPY,
HSI and so on.

1. Compute three moving averages of 20, 40, and 80.
2. Go long when the price crosses above all three moving averages.
3. Exit the long position when the price crosses below any of the three moving
   averages.
4. Go short when the price crosses below all three moving averages.
5. Exit the short position when the price crosses above any of the three moving
   averages.
6. Optional: Optimise all three moving averages

II. Buy and sell the next day

1. Buy the stock on the fourth day open, if the stock closes down consecutively for
   three days.
2. Exit on the next day open.
3. Optional: Optimise the strategy by exiting the long position on the same day close.
   Also, you can optimise the number of down days. There are high chances that the
   number of down days would be different for each stock.

III. Strategy based on RSI indicator.

1. Buy the instrument such as Nifty or SPY when the RSI is less than 15
2. Exit conditions:
   a. Take profit of 5% or RSI > 75  
   b. Stop loss of - 2%
3. Optional: Optimise the strategy by adjusting the RSI value. Also, take profit and stop
   loss criteria can be different for each stock.
4. Note: You can use TA-Lib in Python to compute the RSI value.

IV. Backtest the strategy based on the turtle trading system covered in EFS-02 class. You
can use the daily data for backtesting purpose. Also, the event-driven approach covered in
DMP-01 and DMP-03 would be best for this scenario.

V. Backtest a strategy based on a high & low price

1. Go long when the stock closes above the last 20 day’s high price.
2. Square off the long position when the stock goes below the last 20 day’s low price.
3. Optional: Optimise the strategy by adjusting the number of periods. You can choose
   to have different number of periods for entering the long and exiting the long.

Compute the following statistics for each of the strategy:

1. Compute the buy and hold returns.
2. Compute the strategy returns and compare it with the buy and hold returns.
3. Plot buy and hold returns and strategy returns in a single chart.
4. Plot long and short positions in a single chart.
5. Compute the Sharpe ratio.
6. Compute and plot the drawdown of the strategy.
7. Compute the following:
   a. Number of positive trades
   b. Number of negative trades
   c. Total number of signals generated
   d. Total number of signals traded
   e. Average profit/loss per trade
   f. Hit Ratio
   g. Highest profit & loss in a single trade

You can use the asset of your choice to backtest the strategy. You can use any source of
data that you like. Some of the free data sources are as follows:

1. Pandas-datareader (Allows fetching data from Yahoo, Google and other
   multiple sources)
2. Yfinance (Allows fetching data from Yahoo)
3. Quandl (For major US markets; data might not be free for some data sets)
4. IEXFinance
5. NSEPy (Only for Indian stock markets)
