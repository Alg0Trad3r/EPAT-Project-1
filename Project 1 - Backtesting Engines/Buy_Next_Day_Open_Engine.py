# Importing all necessary libraries to execute strategy
import pandas as pd
import numpy as np
# Importing from personally created modular python file that will compute all of the necessary general statistics 
# and plotting of the strategy
from Strategy_Utility import calc_strategy_returns, plot_returns, bh_returns_return, plot_signals, sharpe_ratio_calc, calc_drawdown, return_plotting_drawdown, plot_max_drawdown, number_of_trades_calc, positive_negative_trades, avg_profit_loss, highest_profit_loss

# I am building this strategy with an Object Oriented Programming (OOP) framework
# This is a strategy that buys the instrument on the 4th day after 3 consecutive
# days where price closed lower than the previous day
# The base strategy will exit the trade at the open of the next day
class Down_Day_Open_Strategy():
  def __init__(self, data_path) -> None:
    self.data_path = data_path
    # Uncomment and run regular function to get base performance of the strategy
    # self.data_handle()
    # self.signal_calc()
    # self.calc_returns()
    # self.drawdown_calc()
    # self.plot_strategy_statistics()
    # self.general_strategy_stats()

    # The optimize function has 3 optimize_param options to choose which
    # result you would like to optimmize for. You must put them in 
    # strings and they are: 'Returns' 'Sharpe Ratio' and 'Drawdown'
    # By passing one of those values to the optimize_param parameter you 
    # will have the function optimize the number of down days for that result
    self.optimize_strategy(target_parameter='Drawdown')

  def data_handle(self):
    # Fetching the path from the passed paramter in the __init__ function  
    data_path = self.data_path
    # Reading the csv file with pandas, setting the dates to the index column,
    # and parsing the dates
    self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # Sorting the dates in ascending order so the plots/graphs represent the data correctly
    self.data = self.data.sort_index(ascending=True)

    # Creating a copy of the original dataframe that we will then work with
    # in the rest of the code
    # This keeps the inital DataFrame untouched so it can be used for other strategies
    self.strategy_df = self.data.copy()
    
    # Debug print
    # print(self.strategy_df.head(6))

  # Calculating the signal of the strategy which is when there are 3 consecutive close days lower than eachother
  # we enter a long position
  def signal_calc(self):
    # Calculating the regular buy and hold returns to be used to calculate if there were 3 negative returns days (This means 
    # there were 3 consecutive days where the close was less than the previous)
    self.strategy_df['B_H_Returns'] = np.round(np.log(self.strategy_df['Close'] / self.strategy_df['Close'].shift(1)), 3)

    # Debug print
    # print(self.strategy_df['B_H_Returns'].head(12))
    # print(self.strategy_df['B_H_Returns'].tail(12))

    # Calculating the signal with a np.where statement that checks if the last 3 days close was less than 0
    # If returns were less than 0 then the close was less than the close the previous day
    self.strategy_df['Long_Signal'] = np.where((self.strategy_df['B_H_Returns'].shift(3) < 0) & (self.strategy_df['B_H_Returns'].shift(2) < 0) & (self.strategy_df['B_H_Returns'].shift(1) < 0), 1, 0)

    # Debug print
    # print(self.strategy_df['B_H_Returns'].head(39))
    # print(self.strategy_df['Long_Signal'].head(39))

    # Calculating the exit of the long which will be a 1 day hold
    # Calculating by checking if the previous day had a long signal and if the close of yesterday does not 
    # equal today (This means it is a new day)
    self.strategy_df['Long_Signal_Exit'] = np.where(self.strategy_df['Long_Signal'].shift(1) == 1 & (self.strategy_df['Close'].shift(1) != self.strategy_df['Close']), 0, self.strategy_df['Long_Signal'])

    # Debug print
    # print(self.strategy_df['Long_Signal_Exit'].head(39))

  # Using a different function than base return calculation function in the Strategy_Stats module because the returns are based
  # on the prices of the open and the close of different days so we need to calculate for those
  def calc_returns(self):
    # Creating columns in the DataFrame that will hold the Open price of the stock when a signal gets triggered and the Open price of
    # the next day when the position is closed
    self.strategy_df['Entry_Price'] = self.strategy_df.loc[self.strategy_df['Long_Signal_Exit'] == 1].Open
    self.strategy_df['Exit_Price'] = self.strategy_df.loc[self.strategy_df['Long_Signal_Exit'] == 0].Open

    # Debug print
    # print(self.strategy_df['Open'].head(27))
    # print(self.strategy_df['Long_Signal_Exit'].head(27))
    # print(self.strategy_df['Entry_Price'].tail(27))
    # print(self.strategy_df['Exit_Price'].tail(27))

    # Calculating the Strategy Returns with the np.log function using the Entry and Exit Price 
    self.strategy_df['Strategy_Returns'] = np.where(self.strategy_df['Entry_Price'] != 0, np.log(self.strategy_df['Entry_Price'] / self.strategy_df['Exit_Price'].shift(1)), 0)
    # Replacing the na values with 0 because if it was not in a trade the returns would be 0 anyways
    # (If you dont do this the total returns will be a NaN value)
    self.strategy_df['Strategy_Returns'] = self.strategy_df['Strategy_Returns'].fillna(0)
    
    # Calculating the cumulative returns for plotting and printing 
    self.cum_strategy_returns = (self.strategy_df['Strategy_Returns'] + 1).cumprod()
    # Printing the total strategy returns for user visual
    print(f'Total Strategy Returns: {np.round(100 * (self.cum_strategy_returns.iloc[-1] - 1), 3)}%')

    # Printing the Buy and Hold returns for user visual
    cum_bh_returns = (self.strategy_df['B_H_Returns'] + 1).cumprod()
    print(f'Total Buy & Hold Returns: {np.round(100 * (cum_bh_returns.iloc[-1] - 1), 3)}%')

    # Debug print
    print(self.strategy_df['Strategy_Returns'].tail(27))
    # Returning the total strategy returns so it can be saved into a variable for future calculations later
    return self.cum_strategy_returns
  
  # Function to calculate the Max Drawdown
  def drawdown_calc(self):
    # Using the function from the modular file I created
    self.max_drawdown = calc_drawdown(strategy_returns=self.strategy_df['Strategy_Returns'])

  # Function that will plot the statistics of the strategy
  def plot_strategy_statistics(self):
    # Creating a variable that holds the return value of the buy and hold returns
    # from the function in the modular file I created
    bh_returns = bh_returns_return(close=self.strategy_df['Close'])
    plot_returns(total_strategy_returns=self.cum_strategy_returns, bh_returns=bh_returns)

    # Plotting the signals (In this case only buy signals)
    # Since signals are only buy signals we will set both of the buy and sell
    # parameters to the Buy Signal in the dataframe
    plot_signals(long_signal=self.strategy_df['Long_Signal_Exit'], short_signal=self.strategy_df['Long_Signal_Exit'])

    # Creating a variable that holds the correct data type that is needed for plotting specifically 
    plot_drawdown = return_plotting_drawdown(strategy_returns=self.strategy_df['Strategy_Returns'])
    # Calling the function from the modular file to plot the drawdown
    plot_max_drawdown(max_drawdown=plot_drawdown)

  # Function that will print the general strategy statistics
  def general_strategy_stats(self):
    # Running the function from my modular file to calculate and print the Sharpe Ratio
    sharpe_ratio_calc(total_returns=self.strategy_df['Strategy_Returns'])

    # Running my function I creaed in my imported modular file to
    # calculate the amount of positive and negative trades
    positive_negative_trades(strategy_returns=self.strategy_df['Strategy_Returns'])

    # Running the function I created to count the number of trades vs number of signals
    # generated from the strategy
    number_of_trades_calc(strategy_trades=self.strategy_df['Long_Signal_Exit'], strategy_signals=self.strategy_df['Long_Signal'])

    # Runing the function I created in my modular file to calculate the
    # average profit/loss per trade
    avg_profit_loss(strategy_returns=self.strategy_df['Strategy_Returns'])

    # Runnng function I created in my modular file to calculate the largest
    # win and loss in a single trade
    highest_profit_loss(strategy_returns=self.strategy_df['Strategy_Returns'])

  # Optimization for this strategy includes recalcuating the exit of the strategy to
  # be on the same day close instead of the next day open
  # Optimization will also include the amount of down days before a long position is entered
  # We will be able to optimize for the Returns, Sharpe Ratio, and Drawdown
  def optimize_strategy(self, target_parameter):
    # Creating a range that will be used for the optimization of the amount of down days
    # needed to take a trade
    down_days_range = range(1, 5)
    
    # Creating a dictionary to store the down days as the key and the target
    # optimization paramter as the pair for optimization
    # A dictionary is created for every target parameter we want to optimize for
    optimized_returns = {}
    optimized_sharpe_ratio = {}
    optimized_drawdown = {}
    
    # Running a for loop to go through the range of down days to see which 
    # amount of down days between 1 and 5 yield the best results 
    for i in down_days_range:
      # We are running the function that will recalculate the signal based off of what 
      # number the amount of down days are
      # This is ran before any of the other logic within the for loop so that the signal will be used
      # for any other functions called that require the signal
      self.optimize_signal_calc(down_days= i)
      # Checking to see if Returns is the paramter we want to optimize for
      if target_parameter == 'Returns':
        # We assign the function that returns the strategy returns to the variable optimize returns
        # so it can be put inside the dictionary and pair with the respective amount of down days 
        # that resulted in that specific calculated return
        optimize_return = self.optimize_returns_calc()
        optimized_returns.update({i: optimize_return})
        # We are pulling the maximum return value out of the dictionary using the the lambda method
        # which allows us to acces the max value and then match it with the correlating key
        self.best_return_pair = max(optimized_returns.items(), key=lambda x: x[1])

      # Using the same process and methodology that was used to optimize the returns to 
      # optimize the sharpe ratio
      if target_parameter == 'Sharpe Ratio':
        self.optimize_returns_calc()
        optimize_sharpe = sharpe_ratio_calc(total_returns=self.strategy_df['Strategy_Returns'])
        optimized_sharpe_ratio.update({i: optimize_sharpe})
        self.best_sharpe_pair = max(optimized_sharpe_ratio.items(), key=lambda x: x[1])

      # Using the same process and methodology that was used to optimize the returns to 
      # optimize the max drawdown
      if target_parameter == 'Drawdown':
        self.optimize_returns_calc()
        optimize_drawdown = calc_drawdown(strategy_returns=self.strategy_df['Strategy_Returns'])
        optimized_drawdown.update({i: optimize_drawdown})
        # We are pulling the max drawdown instead of the minimum drawdown since drawdown is negative we actually
        # want the largest value which is the closest to 0
        self.best_drawdown_pair = max(optimized_drawdown.items(), key=lambda x: x[1])

    # Once paramter has been optimized we check again to make sure that 
    # the returns wanted to be otimized and then run all other stats 
    # using the optimized down days value
    if target_parameter == 'Returns':
      # Saving the list that holds the MA values of the best pair variable
      # by using the [0] index which holds the down days value
      self.best_return_down_days = self.best_return_pair[0]
      i = self.best_return_down_days # Assigning the optimized down days value into
      # its respective variable to do more calculations with them

      print('\n--------- OPTIMIZED RETURNS DOWN DAYS ---------')
      # Printing the best Returns and the key in the dictionary
      print(f'Best Amount Down Days: {self.best_return_pair[0]} -> Highest Return: {np.round(self.best_return_pair[1], 3)}%')
      
      # Running all of the functions we created to calculate signals, get
      # the returns, calculate drawdonw, plotting, and all other general
      # strategy stats with the optimized down days value 
      self.optimize_signal_calc(down_days=i)
      self.optimize_returns_calc()
      self.drawdown_calc()
      self.plot_strategy_statistics()
      self.general_strategy_stats()

    # Following the same logic used to get remaining optimized returns stats
    # to calculate the rest of the stats for the best down days value that give the best Sharpe Ratio
    if target_parameter == 'Sharpe Ratio':
      self.best_sharpe_down_days = self.best_sharpe_pair[0]
      i = self.best_sharpe_down_days

      print('\n--------- OPTIMIZED SHARPE RATIO DOWN DAYS ---------')
      # Printing the best Sharpe Ratio and the key in the dictionary
      print(f'Best Amount Down Days: {self.best_sharpe_pair[0]} -> Best Sharpe Ratio: {np.round(self.best_sharpe_pair[1], 3)}')
      
      # Running all of the functions we created to calculate signals, get
      # the returns, calculate drawdonw, plotting, and all other general
      # strategy stats with the optimized down days value
      self.optimize_signal_calc(down_days=i)
      self.optimize_returns_calc()
      self.drawdown_calc()
      self.plot_strategy_statistics()
      self.general_strategy_stats()

    # Following the same logic used to get remaining optimized returns stats
    # to calculate the rest of the stats for the best down days value that give the best Max Drawdown (Smallest)
    if target_parameter == 'Drawdown':
      self.best_drawdown_down_days = self.best_drawdown_pair[0]
      i = self.best_drawdown_down_days

      print('\n--------- OPTIMIZED MAX DRAWDOWN DOWN DAYS ---------')
      # Printing the best Max Drawdown and the key in the dictionary
      print(f'Best Amount Down Days: {self.best_drawdown_pair[0]} -> Best Max Drawdown: {np.round(self.best_drawdown_pair[1], 3)}%')

      # Running all of the functions we created to calculate signals, get
      # the returns, calculate drawdonw, plotting, and all other general
      # strategy stats with the optimized down days value
      self.optimize_signal_calc(down_days=i)
      self.optimize_returns_calc()
      self.drawdown_calc()
      self.plot_strategy_statistics()
      self.general_strategy_stats()

  # This will be the function that calculates the new signals based off of the optimized
  # parameters ranging from 2 days of negative returns to 5 days (1 week) of negative returns
  def optimize_signal_calc(self, down_days):
    # Running the function to read and handle the data for calculations
    self.data_handle()

    # Calculating the regular buy and hold returns to be used to calculate if there were x number of negative returns days (This means 
    # there were x number of consecutive days where the close was less than the previous)
    self.strategy_df['B_H_Returns'] = np.round(np.log(self.strategy_df['Close'] / self.strategy_df['Close'].shift(1)), 3)

    # Using if statements to check the value of the down days paramter and then recalculating the signal to match whatever the paramter was
    if down_days == 1:
      self.strategy_df['Long_Signal'] = np.where((self.strategy_df['B_H_Returns'].shift(1) < 0), 1, 0)
    elif down_days == 2:
      self.strategy_df['Long_Signal'] = np.where((self.strategy_df['B_H_Returns'].shift(2) < 0) & (self.strategy_df['B_H_Returns'].shift(1) < 0), 1, 0)
    elif down_days == 3:
      self.strategy_df['Long_Signal'] = np.where((self.strategy_df['B_H_Returns'].shift(3) < 0) & (self.strategy_df['B_H_Returns'].shift(2) < 0) & (self.strategy_df['B_H_Returns'].shift(1) < 0), 1, 0)
    elif down_days == 4:
      self.strategy_df['Long_Signal'] = np.where((self.strategy_df['B_H_Returns'].shift(4) < 0) & (self.strategy_df['B_H_Returns'].shift(3) < 0) & (self.strategy_df['B_H_Returns'].shift(2) < 0) & (self.strategy_df['B_H_Returns'].shift(1) < 0), 1, 0)
    elif down_days == 5:
      self.strategy_df['Long_Signal'] = np.where((self.strategy_df['B_H_Returns'].shift(5) < 0) & (self.strategy_df['B_H_Returns'].shift(4) < 0) & (self.strategy_df['B_H_Returns'].shift(3) < 0) & (self.strategy_df['B_H_Returns'].shift(2) < 0) & (self.strategy_df['B_H_Returns'].shift(1) < 0), 1, 0)

    # Calculating the exit of the long which will open the position at the start of the day and exit at the
    # end of the day
    # Calculating by checking if the previous day had a long signal and if the open of today does not 
    # equal the close of today (This means price moved that day)
    self.strategy_df['Long_Signal_Exit'] = np.where(self.strategy_df['Long_Signal'].shift(1) == 1 & (self.strategy_df['Open'] != self.strategy_df['Close']), 0, self.strategy_df['Long_Signal'])

  def optimize_returns_calc(self):
    # Creating columns in the DataFrame that will hold the Open price of the stock when a signal gets triggered and the Open price of
    # the next day when the position is closed
    self.strategy_df['Entry_Price'] = self.strategy_df.loc[self.strategy_df['Long_Signal_Exit'] == 1].Open
    self.strategy_df['Exit_Price'] = self.strategy_df.loc[self.strategy_df['Long_Signal_Exit'] == 0].Close

    # Calculating the Strategy Returns with the np.log function using the Entry and Exit Price 
    self.strategy_df['Strategy_Returns'] = np.where(self.strategy_df['Entry_Price'] != 0, np.log(self.strategy_df['Entry_Price'] / self.strategy_df['Exit_Price'].shift(1)), 0)
    # Replacing the na values with 0 because if it was not in a trade the returns would be 0 anyways
    # (If you dont do this the total returns will be a NaN value)
    self.strategy_df['Strategy_Returns'] = self.strategy_df['Strategy_Returns'].fillna(0)
    
    # Calculating the cumulative returns for plotting and printing 
    self.cum_strategy_returns = (self.strategy_df['Strategy_Returns'] + 1).cumprod()
    total_strategy_returns = 100 * np.round((self.cum_strategy_returns.iloc[-1] - 1), 3)
    # Printing the total strategy returns for user visual
    print(f'Total Strategy Returns: {np.round(100 * (self.cum_strategy_returns.iloc[-1] - 1), 3)}%')

    # Printing the Buy and Hold returns for user visual
    cum_bh_returns = (self.strategy_df['B_H_Returns'] + 1).cumprod()
    print(f'Total Buy & Hold Returns: {np.round(100 * (cum_bh_returns.iloc[-1] - 1), 3)}%')

    # Debug print
    # print(self.strategy_df['Strategy_Returns'].tail(27))

    # Returning the total strategy returns so it can be saved into a variable for future calculations later
    return total_strategy_returns


# Creating datapath that is linked to data saved locally on machine
data_path = "C:/Users/enlig/OneDrive/Desktop/Midas Quant/EPAT/Projects/EPAT-Project-Repositories/EPAT-Project-1/Project 1 - Backtesting Strategies/Instrument Data/SPY S & P 500 HistoricalData.csv"
# Creating the instance of the object that accesses the class and runs all of its functions
spy_back_test = Down_Day_Open_Strategy(data_path=data_path)
  
