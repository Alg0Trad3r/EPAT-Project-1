# This is a general Quantitative Statistics functions that will be used across all of the strategies in the project folder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Base chart format
plt.style.use('seaborn-v0_8-darkgrid')

# Function that calculates Buy & Hold and Strategy returns and then prints out the values
def calc_strategy_returns(close, strategy_signals):
  bh_returns = np.log(close / close.shift(1))
  strategy_returns = strategy_signals.shift(1) * bh_returns
  cumulative_strat_returns = (strategy_returns + 1).cumprod()
  cumulative_bh_returns = (bh_returns + 1).cumprod()
  print(f'\nBuy and Hold Returns: {100 * np.round((cumulative_bh_returns.iloc[-1] - 1), 3)}%')
  print(f'Strategy Returns: {100 * np.round((cumulative_strat_returns.iloc[-1] - 1), 3)}%')
  # Debug print
  # print(strategy_returns)
  return strategy_returns

# This is a simple function with the sole purpose of returning the Dataframe of the cumulative returns so it can be used later to plot
def bh_returns_return(close):
  buy_hold_returns = np.log(close / close.shift(1))
  cumulative_bh_returns = (buy_hold_returns + 1).cumprod()
  # Debug print
  # print(cumulative_bh_returns.head())
  return cumulative_bh_returns

# Function that takes the total returns and calculates the maximum drawdown
def calc_drawdown(strategy_returns):
  # Calculate the cumulative returns and drop any NA values
  cumulative_returns = strategy_returns.dropna(inplace= True)
  cumulative_returns = (strategy_returns + 1).cumprod()
  
  # Calculate the running maximum return to use as baseline for finding the maximum volue 
  running_max_return = np.maximum.accumulate(cumulative_returns)
  running_max_return[running_max_return < 1] = 1 # Ensures running max never drops below 0
  running_max_return = running_max_return.dropna() # Dropping NA values from the running max

  # Calculating the drawdown with the running_max_return and the cumulative_returns
  drawdown = (cumulative_returns) / running_max_return - 1
  max_drawdown = (drawdown.min()) * 100
  print(f'The Maximum Drawdown is: {np.round(max_drawdown, 2)}%') # Print statement as information to the user

  # Returning the max drawdown that will be used for drawdown optimization
  return np.round(max_drawdown, 3)

# This function is for returning the drawdown that is able to be plotted 
# (If we were to return the max_drawdown then the plot will not be able
# to plot the chart correctly and an error will occur)
def return_plotting_drawdown(strategy_returns):
  # Calculate the cumulative returns and drop any NA values
  cumulative_returns = strategy_returns.dropna(inplace= True)
  cumulative_returns = (strategy_returns + 1).cumprod()
  
  # Calculate the running maximum return to use as baseline for finding the maximum volue 
  running_max_return = np.maximum.accumulate(cumulative_returns)
  running_max_return[running_max_return < 1] = 1 # Ensures running max never drops below 0
  running_max_return = running_max_return.dropna() # Dropping NA values from the running max

  # Calculating the drawdown with the running_max_return and the cumulative_returns
  drawdown = (cumulative_returns) / running_max_return - 1

  # Returning the drawdown that will be specifically used for plotting
  return drawdown

# This function plots the strategy returns vs the buy & hold returns on a single chart  
def plot_returns(total_strategy_returns, bh_returns): 
  plt.figure(figsize=(15,9)) # Setting the size of the chart
  # Plotting the values which are parameters being sent from the main strategy file that hold the strategy/buy & hold returns
  total_strategy_returns.plot(label='Strategy Returns')
  bh_returns.plot(label='Buy & Hold Returns')
  plt.grid(True)
  plt.title('Strategy Returns vs. Buy & Hold Returns')
  plt.xlabel('Date')
  plt.ylabel('Returns')
  plt.legend(fontsize= 12)
  plt.show() # Showing the plot

# This function will plot the buy and sell signals on 1 graph  
def plot_signals(long_signal, short_signal):
  plt.figure(figsize=(15,9)) # Setting the size of the chart
  # Plotting the values which are parameters being sent from the main strategy file that hold the signals
  long_signal.plot(label='Long Signals')
  short_signal.plot(label='Short Signals')
  plt.grid(True, alpha = 0.6)
  plt.title('Long & Short Signals')
  plt.xlabel('Date')
  plt.ylabel('Signal')
  plt.legend(fontsize= 12)
  plt.show() # Showing the plot

# Plotting the maximum drawdown
def plot_max_drawdown(max_drawdown):
  plt.figure(figsize=(15,9))
  max_drawdown.plot(label= 'Max Drawdown', color='red')
  plt.title('Max Drawdown')
  plt.xlabel('Date')
  plt.ylabel('Returns')
  # Using the .fill_between method to fill in the space in between the points of the drawdown to make it look like the tradition drawdown graph
  plt.fill_between(max_drawdown.index, max_drawdown, color= 'red')
  plt.grid(True, linewidth= 0.6)
  plt.show()

# Function that uses the standard formula to calculate the Sharpe Ratio
def sharpe_ratio_calc(total_returns):
  sharpe_ratio = total_returns.mean() / total_returns.std() * np.sqrt(252)
  print(f'\nThe Yearly Share Ratio is: {np.round(sharpe_ratio, 3)}')
  return sharpe_ratio

# Function that calculates the long, short, and total trades and signals in the dataframe
def number_of_trades_calc(strategy_trades ,strategy_signals):
  # Debug print
  # print(signals.head(30))

  # Calculate the amount of long, short and trades by taking the length of
  # the dataframe of signals which is the passed variabe
  # We then check and see if the value is equal to 1 for longs and -1 
  # for shorts and count the length of that bool returning true
  total_long_trades = len(strategy_trades.loc[strategy_trades.values == 1])
  total_short_trades = len(strategy_trades.loc[strategy_trades.values == -1])
  # Simply add the total amount of long and short trades together
  total_trades = total_long_trades + total_short_trades

  # Calculating out the amount of signals even the ones that were not traded
  total_long_signals = len(strategy_signals.loc[strategy_signals.values == 1])
  total_short_signals = len(strategy_signals.loc[strategy_signals.values == -1])
  total_signals = total_long_signals + total_short_signals
  
  # F Statements to print out the total number of trades and signals
  print(f'\nTotal Number Of Long Trades: {total_long_trades}')
  print(f'Total Number Of Short Trades: {total_short_trades}')
  print(f'Total Number Of Trades: {total_trades}')
  print(f'\nTotal Number Of Long Signals: {total_long_signals}')
  print(f'Total Number Of Short Signals: {total_short_signals}')
  print(f'Total Number Of Signals: {total_signals}')
  # Since you cannot divide by 0 we are checking to see if the signal count is greater than 0
  # before calculating the percent of trades taken out of the total amount of signals for long, short, and overall
  if total_long_signals > 0:
    print(f'\nPercent Of Long Trades Taken Out Of Total Long Signals: {100 * np.round(total_long_trades / total_long_signals, 2)}%')
  if total_short_signals > 0:
    print(f'Percent Of Short Trades Taken Out Of Total Short Signals: {100 * np.round(total_short_trades / total_short_signals, 2)}%')
  if total_signals > 0:
    print(f'Percent Of Total Trades Taken Out Of Total Signals: {100 * np.round(total_trades / total_signals, 2)}%')

# Function to count the amount of positive or negative trades with the parameter of the strategy returns
def positive_negative_trades(strategy_returns):
  # These linds count the amount of times that the returns where either 
  # positive or negative and saves them in their respective variables
  total_positive_trades = len(strategy_returns.loc[strategy_returns.values > 0])
  total_negative_trades = len(strategy_returns.loc[strategy_returns.values < 0])
  total_trades = total_positive_trades + total_negative_trades

  # Calculating the hit ratio by dividing the amount of positive trades by
  # the total number of trades
  hit_ratio = total_positive_trades / (total_positive_trades + total_negative_trades)

  # Calculate the Win and Loss percentage by dividing the winning/losing trades
  # by the total amount of trades
  win_percentage = (np.round(total_positive_trades / total_trades, 2) * 100) 
  loss_percentage = (np.round(total_negative_trades / total_trades, 2)* 100)
  
  # Print statements to visually see the number of positive and negative trades and Hit Ratio
  print(f'\nTotal Number Of Positive Trades: {total_positive_trades}')
  print(f'Total Number Of Negative Trades: {total_negative_trades}')
  print(f'\nWin Percentage: {win_percentage}%')
  print(f'Loss Percentage: {loss_percentage}%')
  print(f'\nHit Ratio: {np.round(hit_ratio, 3)}')
  return hit_ratio

# Function to calculate the average profit/loss per trade
def avg_profit_loss(strategy_returns):
  avg_profit_per_loss = strategy_returns.loc[strategy_returns != 0].mean()

  # Printing statement to visually show the average profit per average loss on total strategy returns
  print(f'\nAverage Profit/Loss Per Trade: {100 * np.round(avg_profit_per_loss, 3)}%')
  return avg_profit_per_loss

# Function to calculate the highest profit and loss in a single trade
def highest_profit_loss(strategy_returns):
  # Use the same logic/idea that was used to calculate the Average Profit
  # Per Loss metric but applied to the Highest and Lowest Profits Per Trade
  highest_profit = strategy_returns.loc[strategy_returns > 0].max()
  highest_loss = strategy_returns.loc[strategy_returns < 0].min()

  # F Statements to print for the user to visually see
  print(f'\nHighest Profit In Single Trade: {100 * np.round(highest_profit, 3)}%')
  print(f'Highest Loss In Single Trade: {100 * np.round(highest_loss, 3)}%')



  



 




