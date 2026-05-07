# Importing all of the necessary libraries
import numpy as np
# Importing all of the necessary utility and strategy statistics functions from 
# modular file that was created
from Strategy_Utility import data_handler, plot_returns, plot_signals, sharpe_ratio_calc, calc_drawdown, return_plotting_drawdown, plot_max_drawdown, number_of_trades_calc, positive_negative_trades, avg_profit_loss, highest_profit_loss, advanced_strategy_statistics

# This is an object oriented based backtesting engine that will buy the instrument when
# the Open, High or Close breaks above the 20 day rolling High
# This object can also optimize for the user's target parameter
class High_Breakout_Strategy():
  # The innit function that will run all of the functions in
  # the class on whatever instance of the object exists
  def __init__(self, data_path, instrument_name):
    # Setting the data path and instrument names to variables that
    # can be used inside of the class
    self.data_path = data_path
    self.instrument_name = instrument_name

    # Uncomment all of these functions to run the strategy with
    # the base logic
    # self.data_handle()
    # self.signal_calc()
    # self.returns_calc()
    # self.plot_strategy_stats()
    # self.general_strategy_stats()
    # self.advanced_strategy_stats(view_advanced_statistics='Yes')
    
    # This is a function that will run an optimization based off of the 
    # value passed in the target parameter paramter
    # The options for target parameters are only to be formatted in a
    # string and are 'Returns', 'Sharpe', 'Drawdown'
    self.optimize_breakout_strategy(target_parameter='Returns')

  # Function of the class that handles and correctly formats the strategy data
  def data_handle(self):
    # Using the data_handler() function from the utilities module to return
    # the formatted datafraem into the variable strategy_df
    # Making this variable accessibile throughout the entire class with the self. syntax
    self.strategy_df = data_handler(data_path=self.data_path)

    # Debug print
    # print(self.strategy_df.head(50))
    # print(self.strategy_df.tail(50))

  # This function calculates the signals of the strategy
  def signal_calc(self):
    
    # We are first setting our lookback period that will check the
    # rolling high and low of the last 20 periods for entries and exits
    high_breakout_threshold = 20
    low_break_threshold = 20

    # Creating a column in the dataframe that records the 20 day rolling high and low which is 
    # the base strategy instructions
    self.strategy_df['20_Day_High'] = self.strategy_df['High'].rolling(high_breakout_threshold).max().shift(1)
    self.strategy_df['20_Day_Low'] = self.strategy_df['Low'].rolling(low_break_threshold).min().shift(1)

    # Debug print
    # print(self.strategy_df['20_Day_High'].head(50))
    # print(self.strategy_df['20_Day_High'].tail(50))
    # print(self.strategy_df['20_Day_Low'].head(50))
    # print(self.strategy_df['20_Day_Low'].tail(50))


    # To calculate the Entry and Exit triggers we are going to have to use a different
    # method than the classic np.where() function becuase of the fact that
    # the signal needs to be carried across until the exit of the trade

    # To calculate the Entry Trigger we are first going to set a 
    # column in the dataframe equal to a bool that checks if the 
    # Open, High, or Close prices have broken above the 20 Day High
    self.strategy_df['Entry_Trigger'] = ((self.strategy_df['Open'].shift(1) < self.strategy_df['20_Day_High']) & (self.strategy_df['Open'] > self.strategy_df['20_Day_High'])) | ((self.strategy_df['High'].shift(1) < self.strategy_df['20_Day_High']) & (self.strategy_df['High'] > self.strategy_df['20_Day_High'])) | ((self.strategy_df['Close'].shift(1) < self.strategy_df['20_Day_High']) & (self.strategy_df['Close'] > self.strategy_df['20_Day_High']))

    # Similar to how the Entry Trigger was calculated we are calculating the Exit Trigger
    # with a bool that checks if the Open, Low, or Close is less than the 20 Day Low
    self.strategy_df['Exit_Trigger'] = (self.strategy_df['Open'] < self.strategy_df['20_Day_Low']) | (self.strategy_df['Low'] < self.strategy_df['20_Day_Low']) | (self.strategy_df['Close'] < self.strategy_df['20_Day_Low'])

    # To calculate the Long Signal we are creating a column in the strategy dataframe
    # that has a base placeholder of 0
    # We will then create a bool named live_position that is set to false since
    # the strategy will start with no live position
    # Then we will create a list that holds the value of the position signals that is empty
    self.strategy_df['Long_Signal'] = 0
    live_position = False
    position_signals = []

    # To actually calculate the Long Signals we are taking a different approach than
    # the np.where() function 
    # First we are going to first iterate through
    # all of the rows of data in the strategy dataframe 
    for i in range(len(self.strategy_df)):
      # While iterating through all the rows in the strategy dataframe we are
      # going to check if the vale of Entry_Trigger of the current row is True or False with .iloc[i]
      # Since we are not specifying with a == True the compiler automatically reads as 
      # if Entry_Trigger of current row is True
      if self.strategy_df['Entry_Trigger'].iloc[i]:
        # If the Entry Trigger is True then we set the live_position variable to true
        live_position = True
      # If the Exit_Trigger is true of the current row then we are going to
      # set the live_position to false
      elif self.strategy_df['Exit_Trigger'].iloc[i]:
        live_position = False
      # This will append a 1 if the live_position variable is True and 0
      # when it is false to the position_signals list
      position_signals.append(1 if live_position else 0)

    # We then will simply set the column of Long_Signal that just contained 0 to the values
    # that were appended in the position_signals list
    self.strategy_df['Long_Signal'] = position_signals
    
    # Now that the signals have been calculated we need to calculate the Entry and Exit
    # prices so that returns can be calculated

    # To calculate the Entry Price we simply run a np.where() statement to capture the
    # prices where the Long_Signal went from 0 in the last value to 1 --> This means
    # we have entered a trade
    # We set the value when this occurs to the 20 Day High and all of the other times
    # when it does not occur we set the column value to NaN
    self.strategy_df['Trade_Entry_Price'] = np.where((self.strategy_df['Long_Signal'].shift(1) == 0) & (self.strategy_df['Long_Signal'] ==  1), self.strategy_df['20_Day_High'], np.nan)
    # The reason we set the dataframe column value to NaN is becuase we will then use
    # the .ffill() method to extend the last Entry Price all the way down to
    # the next Entry Price
    # Doing this will make it easier to calculate returns
    self.strategy_df['Trade_Entry_Price'] = self.strategy_df['Trade_Entry_Price'].ffill()

    # We are taking the same approach to calculate the Exit Price of the strategy
    # with the np.where() and the .ffill() to create a rolling Exit Price that stays the
    # same until the next Exit Price is calculated
    self.strategy_df['Trade_Exit_Price'] = np.where((self.strategy_df['Long_Signal'].shift(1) == 1) & (self.strategy_df['Exit_Trigger'] == True), self.strategy_df['20_Day_Low'], np.nan)
    self.strategy_df['Trade_Exit_Price'] = self.strategy_df['Trade_Exit_Price'].ffill()

    # Debug print
    # print(self.strategy_df['Long_Signal'].head(50))
    # print(self.strategy_df['Long_Signal'].tail(50))
    # print(self.strategy_df['Trade_Entry_Price'].head(50))
    # print(self.strategy_df['Trade_Entry_Price'].tail(50))
    # print(self.strategy_df['Trade_Exit_Price'].head(50))
    # print(self.strategy_df['Trade_Exit_Price'].tail(50))

  # This function will calculate the returns of the strategy
  # Even though we have a function to calculate the returns in the utility function 
  # we are creating our own since the way to calculate it is a little more nuanced needing more than just
  # the Signals and Buy&Hold Returns
  def returns_calc(self):
    # We are first cleaning up the dataframe of any na values with dropna()
    self.strategy_df.dropna()

    # Calculating the Buy&Hold Returns with a simple np.log() function
    self.strategy_df['BH_Returns'] = np.round(np.log(self.strategy_df['Close'] / self.strategy_df['Close'].shift(1)), 3)

    # To calculate the strategy returns we must check a few things
    # Using the np.where() we pass in a few conditions with the first checking that the Exit Price is not a NaN value
    # It then checks if the current Exit Price is equal to the previous on and if not
    # then it will calculate the returns using the current Exit Price and the previous Entry Price
    # If the current Exit Price is not the same as the previous Exit price this means we have closed a trade
    # This works becuase in the dataframe we rolled all of the exit and entry prices down until
    # the next one was calculated which allows us to reference the previous Entry or Exit 
    # and use it for calculating returns instead of trying to find where the Entry of the previous
    # signal is by looking back in the dataframe
    self.strategy_df['Strategy_Returns'] = np.where((self.strategy_df['Trade_Exit_Price'] != np.nan) & (self.strategy_df['Trade_Exit_Price'] != self.strategy_df['Trade_Exit_Price'].shift(1)), np.round(np.log(self.strategy_df['Trade_Exit_Price'] / self.strategy_df['Trade_Entry_Price'].shift(1)), 3), 0)

    # Debug print
    # print(self.strategy_df['BH_Returns'].head())
    # print(self.strategy_df['BH_Returns'].tail())
    # print(self.strategy_df['Strategy_Returns'].head(50))
    # print(self.strategy_df['Strategy_Returns'].tail(50))

    # We calculate the Cumulative Returns for both B&H and the Strategy with cumprod() 
    # since we are using log returns
    self.strategy_df['Cumulative_BH_Returns'] = (1 + self.strategy_df['BH_Returns']).cumprod()
    self.strategy_df['Cumulative_Strategy_Returns'] = (1 + self.strategy_df['Strategy_Returns']).cumprod()

    # To get the final returns we must print the last value of the column that holds the Cumulative Returns
    # We get that last value with the .iloc[-1] and use a np.round() method to round it to the nearest value of choice
    print(f'\n{self.instrument_name} Total Buy and Hold Returns: {100 * (np.round(self.strategy_df['Cumulative_BH_Returns'].iloc[-1] - 1, 3))}%')
    print(f'{self.instrument_name} Total Strategy Returns: {100 * (np.round(self.strategy_df['Cumulative_Strategy_Returns'].iloc[-1] - 1, 3))}%\n')

    # To be used in optimization we then return the value of the returns so it can be assigned to
    # a variable and used for further calculation
    return 100 * (np.round(self.strategy_df['Cumulative_Strategy_Returns'].iloc[-1] - 1, 3))

  # Creating a function that uses the calc_drawdown() function from the Strategy_Utility module
  # to calculate strategy drawdown
  def calculate_drawdown(self):
    calc_drawdown(strategy_returns=self.strategy_df['Strategy_Returns'], instrument_name=self.instrument_name)

  # Creating a function that will plot all strategy statistics
  def plot_strategy_stats(self):
    # Calling the plot_returns function that willt ake the Cumulative Strategy and B&H
    # returns to plot them
    plot_returns(total_strategy_returns=self.strategy_df['Cumulative_Strategy_Returns'], bh_returns=self.strategy_df['Cumulative_BH_Returns'], instrument_name=self.instrument_name)

    # Plotting the signals (In this case only buy signals)
    # Since signals are only buy signals we will set both of the buy and sell
    # parameters to the Buy Signal in the dataframe
    plot_signals(long_signal=self.strategy_df['Long_Signal'], short_signal=self.strategy_df['Long_Signal'], instrument_name=self.instrument_name)
    
    # Creating a variable that holds the correct data type that is needed for plotting the drawdown specifically 
    strategy_drawdown = return_plotting_drawdown(strategy_returns=self.strategy_df['Strategy_Returns'])
    # Calling the function from the modular file to plot the drawdown
    plot_max_drawdown(max_drawdown=strategy_drawdown, instrument_name=self.instrument_name)

  # Function that will print the general strategy statistics
  def general_strategy_stats(self):
    # Running the function from my modular file to calculate and print the Sharpe Ratio
    sharpe_ratio_calc(total_returns=self.strategy_df['Strategy_Returns'], instrument_name=self.instrument_name)

    # To calculate the number of actual trades that were taken we are going to check when 
    # the Returns were not 0 and if they werent then we just put a 1 there
    # This will be used as a paramter to check the amount of actual trades the strategy executed
    self.strategy_df['Number_of_Trades'] = np.where(self.strategy_df['Strategy_Returns'] != 0, 1, 0)
    # Running the function I created to count the number of trades vs number of signals
    # generated from the strategy
    number_of_trades_calc(strategy_signals=self.strategy_df['Long_Signal'], strategy_trades=self.strategy_df['Number_of_Trades'], instrument_name=self.instrument_name)

    # Running my function I created in my imported modular file to
    # calculate the amount of positive and negative trades
    positive_negative_trades(strategy_returns=self.strategy_df['Strategy_Returns'], instrument_name=self.instrument_name)

    # Runing the function I created in my modular file to calculate the
    # average profit/loss per trade
    avg_profit_loss(strategy_returns=self.strategy_df['Strategy_Returns'], instrument_name=self.instrument_name)

    # Runnng function I created in my modular file to calculate the largest
    # win and loss in a single trade
    highest_profit_loss(strategy_returns=self.strategy_df['Strategy_Returns'], instrument_name=self.instrument_name)

  # Creating a function that calls the advanced_strategy_statistics() function to give advanced statistics view
  # The paramter passed will check if you want to actually see them or not --> to 
  # see the advanced statistics set the parameter to 'Yes' in a string
  def advanced_strategy_stats(self, view_advanced_statistics):
    # Checking to see if the user wants to view the advanced statistics
    if view_advanced_statistics == 'Yes':
      # Cleaning the dataframe by filling in any na values with 0
      self.strategy_df['Strategy_Returns'].fillna(0)
      # Calling function from modular file that will use pyfolio to generate the 
      # advanced statistics charts
      advanced_strategy_statistics(strategy_returns=self.strategy_df['Strategy_Returns'], instrument_name=self.instrument_name)
  # Function created to optimize the user specified parameter of choice
  def optimize_breakout_strategy(self, target_parameter):
    # Setting the ranges of the values that the strategy will run through to
    # optimize for the certain parameters
    high_breakout_threshold = range(3, 60)
    low_breakout_threshold = range(3, 60)

    # Creating the empty dictionaries that will hold the keys and values
    # of the different parameters and target parameter calculated from them
    # that we are optimizing for
    optimized_returns = {}
    optimized_sharpe = {}
    optimized_drawdown = {}

    # Creating for loops that iterate through all of the different combinations of
    # the paramters that will then be used to run the rest of the calculations accordingly
    for i in high_breakout_threshold:
      for j in low_breakout_threshold:
        
        # Creating an check to see which target parameter we are optimizing for and if
        # so then we will run some calculations
        if target_parameter == 'Returns':
          # We will first calculate the new signal with the function created to calculate the 
          # signal with the optimized parameters passing through all of the combinations that
          # are being tested for the calculation values
          self.optimize_signal_calc(i=i, j=j)
          # We are then saving the return of the strategy returns into a variable to then be
          # used to find the best return producing combination of paramters

          # We are then saving the return of the strategy returns into a variable to then be
          # used to find the best return producing combination of paramters
          self.optimize_return = self.returns_calc()

          # We are updating the dictionary with the values of the all of the combinations
          # of optimized parameters as the key and setting the value to the optimize return
          # variable which is just the return of the returns_calc() function
          optimized_returns.update({(i, j): self.optimize_return})
          # Getting the highest return by using the max() function
          # I am setting the key to the lambda function and pulling the value of x[1] 
          # because x[1] is the value associated with the key that is pulled from the 
          # dictionary that holds all of the optimized return key value pairs
          self.best_returns_pair = max(optimized_returns.items(), key=lambda x:x[1])

        # Using the same logic/approach that was used to optimzied for the returns to optimize
        # for the Sharpe Ratio
        elif target_parameter == 'Sharpe':
          self.optimize_signal_calc(i=i, j=j)
          self.returns_calc()
          self.optimize_sharpe_ratio = sharpe_ratio_calc(total_returns=self.strategy_df['Strategy_Returns'], instrument_name=self.instrument_name)

          optimized_sharpe.update({(i, j): self.optimize_sharpe_ratio})
          self.best_sharpe_pair = max(optimized_sharpe, key=lambda x:x[1])

        # Using the same logic that was used to optimzied for the returns to optimize
        # for the drawdown
        elif target_parameter == 'Drawdown':
          self.optimize_signal_calc(i=i, j=j)
          self.returns_calc()
          self.optimize_drawdown = calc_drawdown(strategy_returns=self.strategy_df['Strategy_Returns'], instrument_name=self.instrument_name)

          optimized_drawdown.update({(i, j): self.optimize_drawdown})
          # To get optimize drawdown we use the max() becuase drawdown is a negative number
          # so in reality the largest number is going to closest to 0 which is what we want
          self.best_drawdown_pair = max(optimized_drawdown, key=lambda x:x[1])

    # Running an if statement once we have gotten out of the for loops checking again
    # to see which target parameter we are optimzing for to calculate the rest of the
    # strategy statistics and do plotting
    if target_parameter == 'Returns':
      # We are setting the variable of the return parameters optimized that will be used to calculate
      # the strategy 
      # We are using the best_returns_pair where the [0] index is the key of the dictionary key value
      # pair that holds all of the specific parameters to calculate the RSI, Signal, Returns, etc.
      self.best_threshold = (self.best_returns_pair[0])
      # Assigning the respective values to their respective variables that will be used to pass
      # to functions used to calculate the rest of the Strategy Statistics
      i, j = self.best_threshold

      # Simple print statement in case you are running this engine on multiple strategies you know
      # which instrument is being tested
      print(f'\n--------- {self.instrument_name} OPTIMIZED RETURNS THRESHOLD VALUES ---------')
      # Printing the best returns and the corresponding key in the dictionary with the [0]
      # holding the key and the [1] holding the value
      print(f'{self.instrument_name}Best Lookback Pair: {self.best_returns_pair[0]} -> Highest Return: {self.best_returns_pair[1]}%')

      # Running all other functions in the Object Class that will plot and give
      # all other strategy statistics using the optimized paramters
      self.optimize_signal_calc(i=i, j=j)
      self.returns_calc()
      self.calculate_drawdown()
      self.plot_strategy_stats()
      print(f'\n--------- {self.instrument_name} OPTIMIZED RETURNS STATISTICS ---------')
      self.general_strategy_stats()
      self.advanced_strategy_stats(view_advanced_statistics='Yes')

    # Using the same logic/approach that was used for the Returns to calculate for the Sharpe Ratio
    elif target_parameter == 'Sharpe':
      self.best_threshold = (self.best_sharpe_pair[0])
      i, j = self.best_threshold

      print(f'\n--------- {self.instrument_name} OPTIMIZED SHARPE RATIO THRESHOLD VALUES ---------')
      # Printing the best returns and the key in the dictionary
      print(f'{self.instrument_name} Best Lookback Pair: {self.best_sharpe_pair[0]} -> Highest Return: {self.best_sharpe_pair[1]}%')
      self.optimize_signal_calc(i=i, j=j)
      self.returns_calc()
      self.calculate_drawdown()
      self.plot_strategy_stats()
      print(f'\n--------- {self.instrument_name} OPTIMIZED SHARPE RATIO STATISTICS ---------')
      self.general_strategy_stats()
      self.advanced_strategy_stats(view_advanced_statistics='Yes')

    # Using the same logic/approach that was used for the Returns to calculate for the Max Drawdown
    elif target_parameter == 'Drawdown':
      self.best_threshold = (self.best_drawdown_pair[0])
      i, j = self.best_threshold

      print(f'\n--------- {self.instrument_name} OPTIMIZED DRAWDOWN THRESHOLD VALUES ---------')
      # Printing the best returns and the key in the dictionary
      print(f'{self.instrument_name} Best Lookback Pair: {self.best_drawdown_pair[0]} -> Highest Return: {self.best_drawdown_pair[1]}%')
      self.optimize_signal_calc(i=i, j=j)
      self.returns_calc()
      self.calculate_drawdown()
      self.plot_strategy_stats()
      print(f'\n--------- {self.instrument_name} OPTIMIZED DRAWDOWN STATISTICS ---------')
      self.general_strategy_stats()
      self.advanced_strategy_stats(view_advanced_statistics='Yes')

  # We are using the skeleton and same approaches as the classic optimize_signal_calc() function
  # We are adding some extra parameters i, j to be used as the values of iteration
  # for the high_breakout_threshold and low_breakout_threshold
  def optimize_signal_calc(self, i, j):
    # We are going to call the data_handle() function so that we can just run the optimization
    # function and everything will calculate as if they are all being ran in the __innit__ function
    self.data_handle()

    # Setting the values to be i and j which will be iterated through in a for loop in the main
    # optimization function optimize_breakout_strategy()
    high_breakout_threshold = i
    low_break_threshold = j

    # Another nuance is that instead of just calling the dataframe columns 20_Day_High and 20_Day_Low
    # We are using a f string and the value of i and j to assign the column name of whatever
    # value i and j is which will be different every time
    # This is done whereever the breakout high and low and called
    self.strategy_df[f'{i}_Day_High'] = self.strategy_df['High'].rolling(high_breakout_threshold).max().shift(1)
    self.strategy_df[f'{j}_Day_Low'] = self.strategy_df['Low'].rolling(low_break_threshold).min().shift(1)

    # Debug print
    # print(self.strategy_df['20_Day_High'].head(50))
    # print(self.strategy_df['20_Day_High'].tail(50))
    # print(self.strategy_df['20_Day_Low'].head(50))
    # print(self.strategy_df['20_Day_Low'].tail(50))

    self.strategy_df['Entry_Trigger'] = ((self.strategy_df['Open'].shift(1) < self.strategy_df[f'{i}_Day_High']) & (self.strategy_df['Open'] > self.strategy_df[f'{i}_Day_High'])) | ((self.strategy_df['High'].shift(1) < self.strategy_df[f'{i}_Day_High']) & (self.strategy_df['High'] > self.strategy_df[f'{i}_Day_High'])) | ((self.strategy_df['Close'].shift(1) < self.strategy_df[f'{i}_Day_High']) & (self.strategy_df['Close'] > self.strategy_df[f'{i}_Day_High']))

    self.strategy_df['Exit_Trigger'] = (self.strategy_df['Open'] < self.strategy_df[f'{j}_Day_Low']) | (self.strategy_df['Low'] < self.strategy_df[f'{j}_Day_Low']) | (self.strategy_df['Close'] < self.strategy_df[f'{j}_Day_Low'])

    self.strategy_df['Long_Signal'] = 0
    live_position = False
    position_signals = []

    # Another nuance about this optimization specific signal caluclation is that k is used as the iterator value
    # instead of i becuase if i is used then it will change the value of i that gets assigned as a parameter and
    # break the rest of the code because it will assign i to the index row that is being iterated through
    # which are very large numbers
    for k in range(len(self.strategy_df)):
      if self.strategy_df['Entry_Trigger'].iloc[k]:
        live_position = True
      elif self.strategy_df['Exit_Trigger'].iloc[k]:
        live_position = False
      position_signals.append(1 if live_position else 0)

    self.strategy_df['Long_Signal'] = position_signals

    self.strategy_df['Trade_Entry_Price'] = np.where((self.strategy_df['Long_Signal'].shift(1) == 0) & (self.strategy_df['Long_Signal'] ==  1), self.strategy_df[f'{i}_Day_High'], np.nan)
    self.strategy_df['Trade_Entry_Price'] = self.strategy_df['Trade_Entry_Price'].ffill()

    self.strategy_df['Trade_Exit_Price'] = np.where((self.strategy_df['Long_Signal'].shift(1) == 1) & (self.strategy_df['Exit_Trigger'] == True), self.strategy_df[f'{j}_Day_Low'], np.nan)
    self.strategy_df['Trade_Exit_Price'] = self.strategy_df['Trade_Exit_Price'].ffill()
    
  
# Creating datapath that is linked to data saved locally on machine
data_path = 'C:/Users/enlig/OneDrive/Desktop/Midas Quant/EPAT/Projects/EPAT-Project-Repositories/EPAT-Project-1/Project 1 - Backtesting Engines/Instrument Data/SPY S & P 500 HistoricalData.csv'
# Creating an object of spy_backtest with the High_Breakout_Strategy() class
# Setting the data_path to the local data on the machine and naming the instrument 'SPY' becuase
# that corresponds to the instrument we are running the backtest on
spy_backtest = High_Breakout_Strategy(data_path=data_path, instrument_name='SPY')