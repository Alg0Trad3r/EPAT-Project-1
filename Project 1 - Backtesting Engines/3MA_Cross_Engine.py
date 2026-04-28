# Importing all of the necessary libraries for the intended code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Base chart format
plt.style.use('seaborn-v0_8-darkgrid')
# Importing personal module that has all the functions that will be used to plot and calculate needed project statistics
from Strategy_Utility import calc_strategy_returns, plot_returns, bh_returns_return, plot_signals, sharpe_ratio_calc, calc_drawdown, return_plotting_drawdown, plot_max_drawdown, number_of_trades_calc, positive_negative_trades, avg_profit_loss, highest_profit_loss

# I am creating this strategy with a professional framework using Object Oriented Programming (OOP)
# This MA crossover strategy will go long when price has crossed above all 3 MA lines then exit when price crosses back below any of the lines
# It will do the opposite for the short --> Go short when price has
# crossed below all 3 MA lines and exits when price crosses back above 
# any of the lines
class MA_Cross_Strategy():
  # Making the __innit_ function that will run all 
  # of the functions in the class and give access 
  # to any class methods
  def __init__(self, data_path):
    self.data_path = data_path
    # Uncomment regular functions to run without Optimization
    # self.data_handling()
    # self.MA_compute()
    # self.plot_ma_lines()
    # self.signal_compute()
    # self.returns_calc()
    # self.strategy_drawdown_calc()
    # self.plot_strategy_statistics()
    # self.general_strategy_stats()

    # The optimize function has 3 optimize_param options to choose which
    # result you would like to optimmize for. You must put them in 
    # strings and they are: 'Returns' 'Sharpe Ratio' and 'Drawdown'
    # By passing one of those values to the optimize_param parameter you 
    # will have the function optimize the MA values for that result
    self.optimize_ma_lengths(optimzie_param='Returns')

  def data_handling(self):
    # This code reads the data from the path and puts it in a dataframe
    self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
    # Sorting the dates in ascending order so the plots/graphs represent the data correctly
    self.data = self.data.sort_index(ascending=True)
    # Making a copy of the original dataframe and saving it into another dataframe to be used directly with the strategy
    # Ensures that the original dataframe can be used for other strategies
    self.strategy_df = self.data.copy()

    # Debug print
    # print(self.strategy_df.head())

    # Creating DataFrame columns that look at the previous price 
    # (Will be used to see if price has crossed over the moving averages)
    self.strategy_df['Open_Prev_Day'] = self.strategy_df.Open.shift(1)
    self.strategy_df['High_Prev_Day'] = self.strategy_df.High.shift(1)
    self.strategy_df['Low_Prev_Day'] = self.strategy_df.Low.shift(1)
    self.strategy_df['Close_Prev_Day'] = self.strategy_df.Close.shift(1)
    
    # Debug print
    # print(self.strategy_df.head())

  def MA_compute(self):
    # Create the lengths for the moving average calculations
    self.ma1_length = 20
    self.ma2_length = 40
    self.ma3_length = 80

    # Create the moving average values themselves using the rolling() and mean() methods
    self.strategy_df['MA_1'] = self.strategy_df['Close'].rolling(window=self.ma1_length, center=False).mean()
    self.strategy_df['MA_2'] = self.strategy_df['Close'].rolling(window=self.ma2_length, center=False).mean()
    self.strategy_df['MA_3'] = self.strategy_df['Close'].rolling(window=self.ma3_length, center=False).mean()

    # Calculate the previous day's values in cased needed or wanted 
    self.strategy_df['MA_1_Prev_Day'] = self.strategy_df['MA_1'].shift(1)
    self.strategy_df['MA_2_Prev_Day'] = self.strategy_df['MA_2'].shift(1)
    self.strategy_df['MA_3_Prev_Day'] = self.strategy_df['MA_3'].shift(1)
    # Cleaning up the DataFrame by dropping the NA values
    self.strategy_df.dropna(inplace=True)
    
    # Debug print
    # print(self.strategy_df.head(30))
    # print(self.strategy_df.tail(36))

  # Function to plot the MA lines on a chart with price
  def plot_ma_lines(self):
    # Creating the MA line variables
    ma_1 = self.strategy_df['MA_1']
    ma_2 = self.strategy_df['MA_2']
    ma_3 = self.strategy_df['MA_3']

    plt.figure(figsize=(15, 9)) # Setting the size of the graph
    # Plotting the lines and the price while giving corresponding labels
    ma_1.plot(label='MA 1')
    ma_2.plot(label='MA 2')
    ma_3.plot(label='MA 3')
    self.strategy_df.Close.plot(label='Price Close')
    # Creating the titles and settings of the graph
    plt.grid(True)
    plt.title('MA Lines On Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(fontsize=12)
    plt.show() # Run the .show() method so the chart actually shows up

  def signal_compute(self):
    # Dropping NA values one more time just to make sure all values can be compared 
    self.strategy_df.dropna(inplace=True)
    
    # Creating a np (Numpy) vectorized if statement that checks and sees if the high or close was less than any 3 of the current day MA values and now it is greater than all of the current day MA values for a long signal
    self.strategy_df['Long_Signal'] = np.where(((((self.strategy_df['High_Prev_Day'] < self.strategy_df['MA_1']) | (self.strategy_df['High_Prev_Day'] < self.strategy_df['MA_2']) | (self.strategy_df['High_Prev_Day'] < self.strategy_df['MA_3'])) | ((self.strategy_df['Close_Prev_Day'] < self.strategy_df['MA_1']) | (self.strategy_df['Close_Prev_Day'] < self.strategy_df['MA_2']) | (self.strategy_df['Close_Prev_Day'] < self.strategy_df['MA_3']))) & (((self.strategy_df['High'] > self.strategy_df['MA_1']) & (self.strategy_df['High'] > self.strategy_df['MA_2']) & (self.strategy_df['High'] > self.strategy_df['MA_3'])) | ((self.strategy_df['Close'] > self.strategy_df['MA_1']) & (self.strategy_df['Close'] > self.strategy_df['MA_2']) & (self.strategy_df['Close'] > self.strategy_df['MA_3'])))), 1, 0)

    # Now checking to see if the open, low, or close crossed below any
    # of the 3 moving averages to then exit the long position
    self.strategy_df['Long_Signal_Exit'] = np.where(self.strategy_df['Long_Signal'].shift(1) == 1 & ((self.strategy_df['Open_Prev_Day'] > self.strategy_df['MA_1']) & (self.strategy_df['Open'] < self.strategy_df['MA_1']) | ((self.strategy_df['Open_Prev_Day'] > self.strategy_df['MA_2']) & self.strategy_df['Open'] < self.strategy_df['MA_2']) | ((self.strategy_df['Open_Prev_Day'] > self.strategy_df['MA_3']) & self.strategy_df['Open'] < self.strategy_df['MA_3'])) |((self.strategy_df['Low_Prev_Day'] > self.strategy_df['MA_1']) & (self.strategy_df['Low'] < self.strategy_df['MA_1']) | ((self.strategy_df['Low_Prev_Day'] > self.strategy_df['MA_2']) & self.strategy_df['Low'] < self.strategy_df['MA_2']) | ((self.strategy_df['Low_Prev_Day'] > self.strategy_df['MA_3']) & self.strategy_df['Low'] < self.strategy_df['MA_3'])) | ((self.strategy_df['Close_Prev_Day'] > self.strategy_df['MA_1']) & (self.strategy_df['Close'] < self.strategy_df['MA_1']) | ((self.strategy_df['Close_Prev_Day'] > self.strategy_df['MA_2']) & self.strategy_df['Close'] < self.strategy_df['MA_2']) | ((self.strategy_df['Close_Prev_Day'] > self.strategy_df['MA_3']) & self.strategy_df['Close'] < self.strategy_df['MA_3'])), 0, self.strategy_df['Long_Signal'])

    # Creating a np (Numpy) vectorized if statement that checks and sees
    # if the low or close was greater than any 3 of the current day MA
    # values and now it is less than all of the current day MA values
    # for a short signal
    self.strategy_df['Short_Signal'] = np.where(((((self.strategy_df['Low_Prev_Day'] > self.strategy_df['MA_1']) | (self.strategy_df['Low_Prev_Day'] > self.strategy_df['MA_2']) | (self.strategy_df['Low_Prev_Day'] > self.strategy_df['MA_3'])) | ((self.strategy_df['Close_Prev_Day'] > self.strategy_df['MA_1']) | (self.strategy_df['Close_Prev_Day'] > self.strategy_df['MA_2']) | (self.strategy_df['Close_Prev_Day'] > self.strategy_df['MA_3']))) & (((self.strategy_df['Low'] < self.strategy_df['MA_1']) & (self.strategy_df['Low'] < self.strategy_df['MA_2']) & (self.strategy_df['Low'] < self.strategy_df['MA_3'])) | ((self.strategy_df['Close'] < self.strategy_df['MA_1']) & (self.strategy_df['Close'] < self.strategy_df['MA_2']) & (self.strategy_df['Close'] < self.strategy_df['MA_3'])))), -1, 0)

    # Now checking to see if the open, high, or close crossed above any
    # of the 3 moving averages to then exit the short position
    self.strategy_df['Short_Signal_Exit'] = np.where(self.strategy_df['Short_Signal'].shift(1) == -1 & ((self.strategy_df['Open_Prev_Day'] < self.strategy_df['MA_1']) & (self.strategy_df['Open'] > self.strategy_df['MA_1']) | ((self.strategy_df['Open_Prev_Day'] < self.strategy_df['MA_2']) & self.strategy_df['Open'] > self.strategy_df['MA_2']) | ((self.strategy_df['Open_Prev_Day'] < self.strategy_df['MA_3']) & self.strategy_df['Open'] > self.strategy_df['MA_3'])) |((self.strategy_df['High_Prev_Day'] < self.strategy_df['MA_1']) & (self.strategy_df['High'] > self.strategy_df['MA_1']) | ((self.strategy_df['High_Prev_Day'] < self.strategy_df['MA_2']) & self.strategy_df['High'] < self.strategy_df['MA_2']) | ((self.strategy_df['High_Prev_Day'] < self.strategy_df['MA_3']) & self.strategy_df['High'] > self.strategy_df['MA_3'])) | ((self.strategy_df['Close_Prev_Day'] < self.strategy_df['MA_1']) & (self.strategy_df['Close'] > self.strategy_df['MA_1']) | ((self.strategy_df['Close_Prev_Day'] < self.strategy_df['MA_2']) & self.strategy_df['Close'] > self.strategy_df['MA_2']) | ((self.strategy_df['Close_Prev_Day'] < self.strategy_df['MA_3']) & self.strategy_df['Close'] > self.strategy_df['MA_3'])), 0, self.strategy_df['Short_Signal'])
    
    # Debug print
    # print(self.strategy_df.head(21))
    # print(self.strategy_df['MA_1_Prev_Day'].head(21))
    # print(self.strategy_df['High_Prev_Day'].head(21))
    # print(self.strategy_df['Low_Prev_Day'].head(21))
    # print(self.strategy_df['Close_Prev_Day'].head(21))
    # print(self.strategy_df['Long_Signal'].head(50))
    # print(self.strategy_df['Long_Signal_Exit'].head(50))
    # print(self.strategy_df['Short_Signal'].tail(50))
    # print(self.strategy_df['Short_Signal_Exit'].tail(50))

  # This function calculates the returns of the long and short positions
  # seperately and then adds them together to get the total cumulative
  # strategy returns
  def returns_calc(self):
    # The function calc_strategy_returns is a function from a imported
    # seperate python module file I created which will both print and
    # return the strategy and buy & hold returns
    long_returns = calc_strategy_returns(close=self.strategy_df['Close'], strategy_signals=self.strategy_df['Long_Signal_Exit'])
    short_returns = calc_strategy_returns(close=self.strategy_df['Close'], strategy_signals=self.strategy_df['Short_Signal_Exit'])

    # This is the calculation of the cumulative returns
    # A Dataframe row is added just to keep track of the returns
    self.strategy_df['Total_Returns'] = long_returns + short_returns
    self.strategy_df['Total_Cum_Returns'] = (self.strategy_df['Total_Returns'] + 1).cumprod()

    # Debug print
    # print(self.strategy_df['Total_Returns'].head(21))
    # print(self.strategy_df['Total_Cum_Returns'].head(21))

    # This grabs the long returns and short returns values and adds them
    # until the last value with .iloc[-1] and cumprod() to get the total
    # cumulative returns of the strategy
    cum_long_returns = (long_returns + 1).cumprod()
    cum_short_returns = (short_returns + 1).cumprod()
    self.total_strategy_returns = np.round(cum_long_returns.iloc[-1] - 1, 2) + np.round(cum_short_returns.iloc[-1] - 1, 2)
    # Rounding and printing the total strategy returns
    self.total_strategy_returns = 100 * np.round(self.total_strategy_returns, 3)
    print(f'Total Strategy Returns: {self.total_strategy_returns}%')
    
    # Returning the value of the cumulative strategy returns for later calculation
    return self.total_strategy_returns

  # Function that uses modular function to calculate strategy drawdown
  def strategy_drawdown_calc(self):
    self.max_drawdown = calc_drawdown(strategy_returns=self.strategy_df['Total_Returns'])

  # This function simply calls on functions that were created in the
  # seperate module file I created
  # It first calls the bh_returns_return that simply returns the
  # cumulative buy and hold returns 
  # It then uses the plot_returns function that will plot out the
  # strategy returns against the buy and hold returns  
  def plot_strategy_statistics(self):
    self.buy_hold_returns = bh_returns_return(close=self.strategy_df['Close'])
    plot_returns(total_strategy_returns=self.strategy_df['Total_Cum_Returns'], bh_returns=self.buy_hold_returns)
    # Using the plot_signals function from personally imported module to plot the strategy signals
    plot_signals(long_signal=self.strategy_df['Long_Signal_Exit'], short_signal=self.strategy_df['Short_Signal_Exit'])
    
    # Creating a variable that holds the correct data type that is needed for plotting specifically
    plotting_drawdown = return_plotting_drawdown(strategy_returns=self.strategy_df['Total_Returns']) 
    # Calling the function in personal module to plot the maximum drawdown
    plot_max_drawdown(max_drawdown=plotting_drawdown)
  
  # Function to get general strategy statistics
  def general_strategy_stats(self):
    # Running my function I created in my imported modular file to find the sharpe ratio
    sharpe_ratio_calc(total_returns=self.strategy_df['Total_Returns'])

    # Running my function I creaed in my imported modular file to
    # calculate the amount of positive and negative trades
    positive_negative_trades(strategy_returns=self.strategy_df['Total_Returns'])
    
    # Adding columns to the DataFrame that hold the total amount of
    # actually taken trades and overall signals that were generated by
    # the strategy
    strategy_trades = self.strategy_df['Short_Signal_Exit'] + self.strategy_df['Long_Signal_Exit'] # Adding together the actually taken trades
    self.strategy_df['Total_Strategy_Trades'] = strategy_trades
    strategy_signals = self.strategy_df['Long_Signal'] + self.strategy_df['Short_Signal'] # Adding together the total amount of signals
    self.strategy_df['Total_Strategy_Signals'] = strategy_signals

    # Debug print
    # print(self.strategy_df['Long_Signal_Exit'].head(50))
    # print(self.strategy_df['Short_Signal_Exit'].head(50))
    # print(self.strategy_df['Total_Strategy_Signals'].head(30))

    # Running the function that I created in my modular file giving it
    # the total signals and trades parameters to calculate the number of
    # total signals and number of taken trades
    number_of_trades_calc(strategy_trades=self.strategy_df['Total_Strategy_Trades'], strategy_signals=self.strategy_df['Total_Strategy_Signals'])

    # Runing the function I created in my modular file to calculate the
    # average profit/loss per trade
    avg_profit_loss(strategy_returns=self.strategy_df['Total_Returns'])

    # Runnng function I created in my modular file to calculate the largest
    # win and loss in a single trade
    highest_profit_loss(strategy_returns=self.strategy_df['Total_Returns'])

  # Strategy to optimize the MA values for various statistics like Returns
  # Drawdown, Sharpe Ratio, etc
  def optimize_ma_lengths(self, optimzie_param):
    # We create ranges that we allow our strategy to optimize
    self.ma1_length = range(10, 30)
    self.ma2_length = range(40, 60)
    self.ma3_length = range(70, 90)

    # Optimized return dictionaries to track MA length pairs specifically 
    # to be able to pull the key value pairs that will hold the best 
    # values optimized
    optimized_returns = {}
    optimized_sharpe = {}
    optimized_drawdown = {}

    # Run for loops that iterate through the ranges of the MA values
    for i in self.ma1_length:
      for j in self.ma2_length:
        for k in self.ma3_length:

          # We are checking to see if the parameter we want to optimize is 
          # Returns and if so we will optimize the MA values for the largest
          # Returns and find the key that assigned to that value
          if optimzie_param == 'Returns':
            # We run a seperate function than the original MA value calculation
            # function to calculate with all of the iterations of the for statements
            self.ma_optimize_calc(i=i, j=j, k=k)
            self.signal_compute() # Calculating the signal with the already 
            # created function
            self.optimize_return = self.returns_calc()
            # We are updating the dictionary with every key value pair
            # that could exist in the for loops to find the maximum value
            # and then matching it with the corresponding key
            optimized_returns.update({(i, j, k): self.optimize_return})
            self.best_pair = max(optimized_returns.items(), key=lambda x: x[1])
          
          # We are checking to see if the parameter that wants to be 
          # optimized for is the Sharpe Ratio
          if optimzie_param == 'Sharpe Ratio':
            # We run a seperate function than the original MA value calculation
            # function to calculate with all of the iterations of the for statements
            self.ma_optimize_calc(i=i, j=j, k=k)
            self.signal_compute() # Calculating the signal with the already 
            # created function
            # Following the same logic that was used for the returns to optimize for the Sharpe Ratio
            self.returns_calc() 
            self.optimize_sharpe_ratio = sharpe_ratio_calc(total_returns=self.strategy_df['Total_Returns'])

            optimized_sharpe.update({(i, j, k): self.optimize_sharpe_ratio})
            self.best_sharpe_pair = max(optimized_sharpe.items(), key=lambda x: x[1])
          
          # Using the same logic to optimize for the Maximum Drawdown as
          # I did for the Sharpe Ratio and Returns
          if optimzie_param == 'Drawdown':
            # We run a seperate function than the original MA value calculation
            # function to calculate with all of the iterations of the for statements
            self.ma_optimize_calc(i=i, j=j, k=k)
            self.signal_compute() # Calculating the signal with the already 
            # created function
            self.returns_calc()
            self.optimize_drawdown = calc_drawdown(strategy_returns=self.strategy_df['Total_Returns'])

            optimized_drawdown.update({(i, j, k): self.optimize_drawdown})
            # Getting the max value of drawdown since drawdown is negative we want the largest value
            self.best_drawdown_pair = max(optimized_drawdown.items(), key=lambda x: x[1])

    # Once paramter has been optimized we check again to make sure that 
    # the returns wanted to be otimized and then run all other stats 
    # using the optimized MA values        
    if optimzie_param == 'Returns':
      # Saving the list that holds the MA values of the best pair variable
      # by using the [0] index which holds the MA values
      self.best_ma = (self.best_pair[0])
      i, j, k = self.best_ma # Assigning the optimized MA values into
      # their respective variables to do more calculations with them
      
      # Debug print
      #print(f'MA 1: {i}, MA 2: {j}, MA 3: {k}')

      print('\n--------- OPTIMIZED RETURNS MA VALUES ---------')
      # Printing the best returns and the key in the dictionary
      print(f'Best MA Pair: {self.best_pair[0]} -> Highest Return: {self.best_pair[1]}%')

      # Running all of the functions we created to calculate signals, get
      # the returns, calculate drawdonw, plotting, and all other general
      # strategy stats
      self.ma_optimize_calc(i=i, j=j, k=k)
      self.signal_compute()
      self.returns_calc()
      self.strategy_drawdown_calc()
      self.plot_strategy_statistics()
      print('\n--------- OPTIMIZED RETURNS STATISTICS ---------')
      self.general_strategy_stats()

    # Following the same logic used to calculate the rest of the stats for
    # the best MA values that give the best Sharpe Ratio
    if optimzie_param == 'Sharpe Ratio':
      self.best_ma_sharpe = (self.best_sharpe_pair[0])
      i, j, k = self.best_ma_sharpe

      print('\n--------- OPTIMIZED SHARPE RATIO MA VALUES ---------')
      # Printing the best Sharpe Ratio and the key in the dictionary
      print(f'The Best MA Values: {self.best_sharpe_pair[0]} -> Best Sharpe Ratio: {np.round(self.best_sharpe_pair[1], 3)}')

      # Running all of the functions we created to calculate signals, get
      # the returns, calculate drawdonw, plotting, and all other general
      # strategy stats
      self.ma_optimize_calc(i=i, j=j, k=k)
      self.signal_compute()
      self.returns_calc()
      self.strategy_drawdown_calc()
      self.plot_strategy_statistics()
      print('\n--------- OPTIMIZED SHARPE RATIO STATISTICS ---------')
      self.general_strategy_stats()

    # Using the same logic used for the best Returns and Sharpe Ratio to
    # calculate all of the other general stats and plots
    if optimzie_param == 'Drawdown':
      self.best_ma_drawdown = (self.best_drawdown_pair[0])
      i, j, k = self.best_ma_drawdown

      print('\n--------- OPTIMIZED DRAWDOWN MA VALUES ---------')
      # Printing the best Sharpe Ratio and the key in the dictionary
      print(f'The Best MA Values: {self.best_drawdown_pair[0]} -> Best Maximum Drawdown: {np.round(self.best_drawdown_pair[1], 3)}')

      # Running all of the functions we created to calculate signals, get
      # the returns, calculate drawdonw, plotting, and all other general
      # strategy stats
      self.ma_optimize_calc(i=i, j=j, k=k)
      self.signal_compute()
      self.returns_calc()
      self.strategy_drawdown_calc()
      self.plot_strategy_statistics()
      print('\n--------- OPTIMIZED MAX DRAWDOWN STATISTICS ---------')
      self.general_strategy_stats()

  def ma_optimize_calc(self, i, j, k):
    # Running the data handling function so Dataframe can be created for calculations
    self.data_handling()
    # Running the function that computes the MA lines on the chart for visual check
    self.MA_compute()

    # Create the moving average values themselves using the rolling() and mean() methods
    self.strategy_df['MA_1'] = self.strategy_df['Close'].rolling(window=i, center=False).mean()
    self.strategy_df['MA_2'] = self.strategy_df['Close'].rolling(window=j, center=False).mean()
    self.strategy_df['MA_3'] = self.strategy_df['Close'].rolling(window=k, center=False).mean()

# Creating datapath that is linked to data saved locally on machine
data_path = 'C:/Users/enlig/OneDrive/Desktop/Midas Quant/EPAT/Projects/EPAT-Project-Repositories/EPAT-Project-1/Project 1 - Backtesting Engines/Instrument Data/SPY S & P 500 HistoricalData.csv'
# Creating the instance of the object that accesses the class and runs all of its functions
spy_test = MA_Cross_Strategy(data_path=data_path)