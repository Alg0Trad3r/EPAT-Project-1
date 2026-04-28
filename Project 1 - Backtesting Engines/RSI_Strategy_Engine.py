# Importing all necessary libraries
import pandas as pd
import numpy as np
import talib as ta
from Strategy_Utility import calc_strategy_returns, plot_returns, bh_returns_return, plot_signals, sharpe_ratio_calc, calc_drawdown, return_plotting_drawdown, plot_max_drawdown, number_of_trades_calc, positive_negative_trades, avg_profit_loss, highest_profit_loss

# This strategy will buy the instrument when the RSI is less than 15
# This strategy will exit at a take profit of 5% or or the RSI is greater than 75
# This strategy will exit with a stop loss of -2%
# Creating OOP based class for the backtesting of the RSI strategy
class RSI_Strategy():
  # The innit function will automatically run any functions called within it
  # on whichever object it is assigned to
  def __init__(self, data_path) :
    self.data_path = data_path
    # Uncomment these functions to run the strategy as its base version with no optimization
    # self.data_handler()
    # self.rsi_calc(rsi_length=14)
    # self.signal_calc()
    # self.strategy_calc_returns()
    # self.strategy_calc_drawdown()
    # self.plot_strategy_statistics()
    # self.strategy_statistics_calc()
    
    # Running only the function that optimizes the parameters for the chosen target parameter
    # To choose between which parameter to optimize for you must assign the target param parameter to the strings 'Returns',
    # 'Sharpe Ratio', and 'Drawdown'
    # WARNING: This optimization depending on the range size of parameters you are setting may take over 30-45
    # minutes to finish running
    self.optimize_strategy_parameters(target_param='Returns')

  # This function handles the organization and initial creation of the DataFrame that will be
  # worked with for the strategy
  def data_handler(self):
    # Reads the path and converts the csv file into a DataFrame
    data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)

    # Sorts the data in an ascending order which is key for Equity Curve and Returns
    # calculations
    data = data.sort_index(ascending=True)

    # Making a copy of the original DataFrame that will be used for the rest of
    # the strategy calculations
    self.strategy_df = data.copy()

    # Debug print
    # print(self.strategy_df.head())

    # Returning the DataFrame that will be used for strategy calculations
    return self.strategy_df

  # This function calculates the RSI values with a user specified length
  def rsi_calc(self, rsi_length):
    # Setting the RSI length in the function to the user defined parameter that will
    # calculate the value of the RSI
    rsi_length = rsi_length
    # The RSI indicator needs an array for its calculations so we are turning
    # the Close column in the DataFRame into an np array in which the RSI indicator
    # will use for calculations
    rsi_data = np.array(self.strategy_df['Close'])
    rsi = ta.RSI(rsi_data, timeperiod=rsi_length) # Calculating the RSI

    # Creating a column in the strategy DataFrame that hold the individual values
    # of the RSI
    self.strategy_df['RSI_Values'] = rsi

    # Dropping NA values to clean up the DataFrame
    self.strategy_df.dropna(inplace=True)

    # Debug print
    # print(self.strategy_df['RSI_Values'].head(50))
    # print(self.strategy_df['RSI_Values'].tail(6))

    # Returning the RSI values in the event we want to work with just those values
    # later in the code
    return self.strategy_df['RSI_Values']
  
  # This will calculate the signal of the strategy
  # The base signal is a long only signal that will go long when the RSI is less than 15
  # The base exit signal is a take profit of 5% or the RSI going above 75 or a stop loss
  # of 3%
  def signal_calc(self):
    # Setting the base parameters of the strategy and saving them into their respective variables
    rsi_entry_threshold = 15
    rsi_exit_threshold = 75
    stop_loss_threshold = -3
    take_profit_threshold = 5

    # Calculating the Buy and Hold Returns along with the Long Signal
    # Using the base strategy rules to calculate the long signal
    # I am calculating the long signal from below the RSI entry trigger all the way until the
    # RSI exit threshold has been reached because this will allow us to calculate the stop loss
    # and take profit
    self.strategy_df['BH_Returns'] = (np.log(self.strategy_df['Close'] / self.strategy_df['Close'].shift(1)) * 100)
    self.strategy_df['Long_Signal'] = np.where((self.strategy_df['RSI_Values'] < rsi_entry_threshold) | (self.strategy_df['RSI_Values'] < rsi_exit_threshold), 1, 0)

    # Calculating the Entry Price for whenever the long signal is equal to 1 and setting it so that it only calculates once
    # If the strategy is alrady in a long it will not continue to calculate a new entry price until the last long signal was a 0
    # and the current long signal is a 1
    # I am setting all of the values other than the initial entry prices as NaN values because they will be getting filled in
    # with the previous Entry Price
    self.strategy_df['Entry_Price'] = np.where((self.strategy_df['Long_Signal'] == 1) & (self.strategy_df['Long_Signal'] != self.strategy_df['Long_Signal'].shift(1)), self.strategy_df['Close'], np.nan)
    # Using the .ffill() method to replace all NaN values with the most recent calculated Entry Price
    # Doing this will allow us to calculate the Take Profit and Stop Loss from the original Entry Price of the position
    self.strategy_df['Entry_Price_Rolling'] = self.strategy_df['Entry_Price'].ffill()

    # Calculating the returns of the open positions by using the Entry Price that is filled in for every row that was a NaN value
    self.strategy_df['Position_Returns'] = np.log(self.strategy_df['Close'] / self.strategy_df['Entry_Price_Rolling'].shift(1)) * 100

    # Calculating the Exit signal that will be used to calculate returns using the Position Returns to check and see if we have hit our
    # stop loss, take profit, or the RSI value has gone over the exit threshold
    self.strategy_df['Long_Exit'] = np.where((((self.strategy_df['Long_Signal'].shift(1) == 1) & (self.strategy_df['RSI_Values'] > rsi_exit_threshold)) | ((self.strategy_df['Long_Signal'].shift(1) == 1) & (self.strategy_df['Position_Returns'] <= stop_loss_threshold)) | ((self.strategy_df['Long_Signal'].shift(1) == 1) & (self.strategy_df['Position_Returns'] >= take_profit_threshold))), 0, self.strategy_df['Long_Signal'])

    # Debug print
    # print(self.strategy_df['BH_Returns'].head(50))
    # print(self.strategy_df['Long_Signal'].head(50))
    # print(self.strategy_df['Entry_Price'].head(50))
    # print(self.strategy_df['Entry_Price_Rolling'].head(50))
    # print(self.strategy_df['Position_Returns'].head(50))
    # print(self.strategy_df['Long_Exit'].head(50))

  # Function to calculate the strategy returns 
  def strategy_calc_returns(self):
    # Assigning the returned values from the function to calculate strategy returns into a column
    # in the strategy DataFrame 
    self.strategy_df['Strategy_Position_Returns'] = calc_strategy_returns(close=self.strategy_df['Close'], strategy_signals=self.strategy_df['Long_Exit'])
    # Calculating the Buy and Hold Returns using the function from the modular Strategy_Stats file
    self.bh_returns = bh_returns_return(close=self.strategy_df['Close']) 

    # Calculating the cumulative returns using the column created in the Strategy DataFrame
    self.cumulative_returns = (self.strategy_df['Strategy_Position_Returns'] + 1).cumprod()

    # Debug print
    # print(self.strategy_df['Strategy_Position_Returns'].head(50))

    # Calculating the total strategy returns by grabbing the last value in the cumulative returns
    # variable after the .cumprod() method has calculated all the total returns
    total_strategy_returns = np.round(self.cumulative_returns.iloc[-1] - 1 , 3) * 100

    # Returning the total returns of the strategy so it can be saved into a variable and used
    # used in further calculation specifically when going to optimize parameters
    return total_strategy_returns
  
  # Creating a function to calculate the strategy drawdowns
  # Using the function from the Strategy Stats modular file to calculate them 
  def strategy_calc_drawdown(self):
    self.strategy_drawdown = calc_drawdown(strategy_returns=self.strategy_df['Strategy_Position_Returns'])

  # Function to plot the strategy metrics and statistics
  def plot_strategy_statistics(self):
    # Using the function from the modular Strategy Stats file to plot returns
    plot_returns(total_strategy_returns=self.cumulative_returns, bh_returns=self.bh_returns)

    # Using the function from the modular Strategy Stats file to plot the signals
    # Since there are only long signals we just assign both paramters to the Long Exit column 
    # of the Strategy DataFrame
    plot_signals(long_signal=self.strategy_df['Long_Exit'], short_signal=self.strategy_df['Long_Exit'])

    # Using the function from the modular Strategy Stats file to plot drawdown
    drawdown_plot = return_plotting_drawdown(strategy_returns=self.strategy_df['Strategy_Position_Returns'])
    plot_max_drawdown(max_drawdown=drawdown_plot)

  # Function to calculate all other necessary statistics for the strategy
  def strategy_statistics_calc(self):
    # Using the function from the modular Strategy Stats file to calculate the sharpe ratio
    sharpe_ratio_calc(total_returns=self.strategy_df['Strategy_Position_Returns'])

    # Using the function from the modular Strategy Stats file to calculate the varios trade
    # metrics such as the total number of trades relative to signals
    number_of_trades_calc(strategy_trades=self.strategy_df['Long_Exit'], strategy_signals=self.strategy_df['Long_Signal'])

    # Using the function from the modular Strategy Stats file to calculate the metrics such as 
    # the hit ratio, positive(profitable trades), negative(loosing trades), win, and loss percentage
    positive_negative_trades(strategy_returns=self.strategy_df['Strategy_Position_Returns'])

    # Using the function from the modular Strategy Stats file to calculate the average profit/loss
    avg_profit_loss(strategy_returns=self.strategy_df['Strategy_Position_Returns'])

    # Using the function from the modular Strategy Stats file to calculate the highest profit
    # and loss per single trade
    # Note it is only using the close prices for the returns so values may be larger than
    # stop loss/take profit threshold because on some days the returns of the next day exceeded the
    # threshold so the trade was closed but the returns were still larger
    highest_profit_loss(strategy_returns=self.strategy_df['Strategy_Position_Returns'])

  # Function created to optimize the user specified parameter of choice 
  def optimize_strategy_parameters(self, target_param):
    # Simple print statement to show that the optimization has started
    print('\n--------- OPTIMIZATION OF PARAMETER VALUES INITIATED ---------')
    # Setting the ranges of the values that the strategy will run through to
    # optimize for the certain parameters
    self.rsi_length = range(2, 25)
    self.rsi_entry_threshold = range(12, 45)
    self.rsi_exit_threshold = range(36, 81)
    self.stop_loss_threshold = range(1, 3)
    self.take_profit_threshold = range(1, 9)

    # Creating the empty dictionaries that will hold the keys and values
    # of the different parameters and target parameter calculated from them
    # that we are optimizing for
    optimized_returns = {}
    optimized_sharpe_ratio = {}
    optimized_drawdown = {}

    # Creating for loops that iterate through all of the different combinations of
    # the paramters that will then be used to run the rest of the calculations accordingly
    for i in self.rsi_length:
      for j in self.rsi_entry_threshold:
        for k in self.rsi_exit_threshold:
          for l in self.stop_loss_threshold:
            for m in self.take_profit_threshold:

              # Creating an check to see which target parameter we are optimizing for and if
              # so then we will run some calculations
              if target_param == 'Returns':
                # We will first calculate the new signal with a function created to calculate the 
                # signal with the optimized parameters passing through all of the combinations that
                # are being tested 
                self.optimize_signal_calc(i=i, j=j, k=k, l=-l, m=m)
                # We are then saving the return of the strategy returns into a variable to then be
                # used to find the best return producing combination of paramters
                self.optimize_return = self.strategy_calc_returns()
                
                # This is a check that will only save the returns into the dictionary meant
                # for optimized returns that are greater than 0
                if self.optimize_return > 0:
                  # We are updating the dictionary with the values of the all of the combinations
                  # of optimized parameters as the key and setting the value to the optimize return
                  # variable which is just the return of the strategy_calc_returns() function
                  optimized_returns.update({(i, j, k, l, m): self.optimize_return})
                  # Getting the highest return by using the max() function
                  # I am setting the key to the lambda function and pulling the value of x[1] 
                  # because x[1] is the value associated with the key that is pulled from the 
                  # dictionary that holds all of the optimized return key value pairs
                  self.best_returns_pair = max(optimized_returns.items(), key=lambda x:x[1])
                  
                  # Debug print
                  # print(f'Returns: {self.optimize_return}')
                  # print(self.best_returns_pair)

                # Using the same logic that was used to optimzied for the returns to optimize
                # for the Sharpe Ratio
                if target_param == 'Sharpe Ratio':
                  self.optimize_signal_calc(i=i, j=j, k=k, l=-l, m=m)
                  returns = self.strategy_calc_returns()
                  self.optimize_sharpe = sharpe_ratio_calc(total_returns=self.strategy_df['Strategy_Position_Returns'])
                  
                  if returns > 0:
                    optimized_sharpe_ratio.update({(i, j, k, l, m): self.optimize_sharpe})
                    self.best_sharpe_pair = max(optimized_returns.items(), key=lambda x:x[1])

                # Using the same logic that was used to optimzied for the returns to optimize
                # for the drawdown
                if target_param == 'Drawdown':
                  self.optimize_signal_calc(i=i, j=j, k=k, l=-l, m=m)
                  returns = self.strategy_calc_returns()
                  self.optimize_drawdown = calc_drawdown(strategy_returns=self.strategy_df['Strategy_Position_Returns'])

                  if returns > 0:
                    optimized_drawdown.update({(i, j, k, l, m): self.optimize_drawdown})
                    self.best_drawdown_pair = max(optimized_drawdown.items(), key=lambda x:x[1])

    # Running an if statement once we have gotten out of the for loops checking again
    # to see which target parameter we are optimzing for to calculate the rest of the
    # strategy statistics and do plotting
    if target_param == 'Returns':
      # We are setting the variable of the return parameters optimized that will be used to calculate
      # the strategy 
      # We are using the best_returns_pair where the [0] index is the key of the dictionary key value
      # pair that holds all of the specific parameters to calculate the RSI, Signal, Returns, etc.
      self.best_returns_params = (self.best_returns_pair[0])

      # Assigning the respective values to their respective variables that will be used to pass
      # to functions used to calculate the rest of the Strategy Statistics
      i, j, k, l, m = self.best_returns_params 

      print('\n--------- OPTIMIZED RETURNS PARAMETER VALUES ---------')
      # Printing the best returns and the key in the dictionary using the values assigned from the best_returns_params
      # The [0] index is the key values and the [1] index is the actual value(Returns)
      print(f'RSI Length: {i},\nRSI Entry Threshold: {j}, RSI Exit Threshold: {k},\nStop Loss Threshold: {-l}, Take Profit Threshold: {m}')
      print(f'Best Parameters Pair: {self.best_returns_pair[0]} -> Highest Return: {self.best_returns_pair[1]}%')

      # Running all other functions in the Object Class that will plot and give
      # all other strategy statistics using the optimized paramters
      self.optimize_signal_calc(i=i, j=j, k=k, l=-l, m=m)
      self.strategy_calc_returns()
      self.strategy_calc_drawdown()
      self.plot_strategy_statistics()
      print('\n--------- OPTIMIZED RETURNS STATISTICS ---------')
      self.strategy_statistics_calc()

    # Using the same logic that was used for the Returns to calculate for the Sharpe Ratio
    if target_param == 'Sharpe Ratio':
      self.best_sharpe_params = (self.best_sharpe_pair[0])

      i, j, k, l, m = self.best_sharpe_params 

      print('\n--------- OPTIMIZED SHARPE PARAMETER VALUES ---------')
      # Printing the best returns and the key in the dictionary
      print(f'RSI Length: {i},\nRSI Entry Threshold: {j}, RSI Exit Threshold: {k},\nStop Loss Threshold: {-l}, Take Profit Threshold: {m}')
      print(f'Best Parameters Pair: {self.best_sharpe_pair[0]} -> Highest Return: {self.best_sharpe_pair[1]}%')

      self.optimize_signal_calc(i=i, j=j, k=k, l=-l, m=m)
      self.strategy_calc_returns()
      self.strategy_calc_drawdown()
      self.plot_strategy_statistics()
      print('\n--------- OPTIMIZED SHARPE RATIO STATISTICS ---------')
      self.strategy_statistics_calc()

    # Using the same logic that was used for the Returns to calculate for the Max Drawdown
    if target_param == 'Drawdown':
      self.best_drawdown_params = (self.best_drawdown_pair[0])

      i, j, k, l, m = self.best_drawdown_params 

      print('\n--------- OPTIMIZED DRAWDOWN PARAMETER VALUES ---------')
      # Printing the best returns and the key in the dictionary
      print(f'RSI Length: {i},\nRSI Entry Threshold: {j}, RSI Exit Threshold: {k},\nStop Loss Threshold: {-l}, Take Profit Threshold: {m}')
      print(f'Best Parameter Pair: {self.best_drawdown_pair[0]} -> Highest Return: {self.best_drawdown_pair[1]}%')

      self.optimize_signal_calc(i=i, j=j, k=k, l=-l, m=m)
      self.strategy_calc_returns()
      self.strategy_calc_drawdown()
      self.plot_strategy_statistics()
      print('\n--------- OPTIMIZED DRAWDOWN STATISTICS ---------')
      self.strategy_statistics_calc()

  # Creating a seperate function that is the same as the other signal_calc() function with
  # the only difference that it will take the parameters of all of the different parameters
  # that can be optimized for
  def optimize_paramters_signal_calc(self, rsi_entry_threshold, rsi_exit_threshold, stop_loss_threshold, take_profit_threshold):

    self.strategy_df['BH_Returns'] = (np.log(self.strategy_df['Close'] / self.strategy_df['Close'].shift(1)) * 100)
    self.strategy_df['Long_Signal'] = np.where((self.strategy_df['RSI_Values'] < rsi_entry_threshold) | (self.strategy_df['RSI_Values'] < rsi_exit_threshold), 1, 0)

    # self.strategy_df['Long_Signal'] = np.where((self.strategy_df['Long_Trigger'] == np.nan) & (self.strategy_df['Long_Trigger'] != self.strategy_df['Long_Trigger'].shift(1)) , self.strategy_df['Long_Trigger'].shift(1), self.strategy_df['Long_Trigger'].ffill())

    self.strategy_df['Entry_Price'] = np.where((self.strategy_df['Long_Signal'] == 1) & (self.strategy_df['Long_Signal'] != self.strategy_df['Long_Signal'].shift(1)), self.strategy_df['Close'], np.nan)
    self.strategy_df['Entry_Price_Rolling'] = self.strategy_df['Entry_Price'].ffill()

    self.strategy_df['Position_Returns'] = np.log(self.strategy_df['Close'] / self.strategy_df['Entry_Price_Rolling'].shift(1)) * 100

    self.strategy_df['Long_Exit'] = np.where((((self.strategy_df['Long_Signal'].shift(1) == 1) & (self.strategy_df['RSI_Values'] > rsi_exit_threshold)) | ((self.strategy_df['Long_Signal'].shift(1) == 1) & (self.strategy_df['Position_Returns'] <= stop_loss_threshold)) | ((self.strategy_df['Long_Signal'].shift(1) == 1) & (self.strategy_df['Position_Returns'] >= take_profit_threshold))), 0, self.strategy_df['Long_Signal'])

    # Debug print
    # print(self.strategy_df['Long_Exit'].head(30))
    # print(self.strategy_df['Long_Exit'].tail(30))

  # This function will run a few functions to ultimately calculate the signal
  # It takes the parameters that reflect the iterations that happen in the for
  # loops that are used for the optimization in the optimize_strategy_params() function
  def optimize_signal_calc(self, i, j, k, l, m):
    # We must first prepare the data using the data handling function from the Object Class
    self.data_handler()

    # Using the RSI value calculation function from the Object Class while passing through the corresponding
    # variable of the RSI length which is i in the optimize_strategy_params() function
    self.rsi_calc(rsi_length=i)

    # Running the function that will calculate the signal with all of the values that are being optimized 
    # for in the optimize_strategy_params() function
    self.optimize_paramters_signal_calc(rsi_entry_threshold=j, rsi_exit_threshold=k, stop_loss_threshold=-l, take_profit_threshold=m)


# Creating datapath that is linked to data saved locally on machine
data_path = "C:/Users/enlig/OneDrive/Desktop/Midas Quant/EPAT/Projects/EPAT-Project-Repositories/EPAT-Project-1/Project 1 - Backtesting Strategies/Instrument Data/SPY S & P 500 HistoricalData.csv"
# Creating the instance of the object that accesses the class and runs all of its functions
spy_test = RSI_Strategy(data_path=data_path)