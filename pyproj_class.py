import requests #Requests allows you to send HTTP/1.1 requests
import json #JSON is a syntax for storing and exchanging data, the data we'll get from Binance's API are in this sintax format
import pandas as pd
import datetime as dt 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings #This Library is used because at LINE there was a warning regarding the use of an old function, 
warnings.filterwarnings('ignore') #we also tried to use the new one but was not supported in our model


class DATA:
    def __init__(self,symbol,startTime,endTime): #initializing the class in relation to the crypto ticker (symbol) from which we want to import hystorical data and anlso for the start and end time
        self.symbol=symbol
        self.startTime = startTime
        self.endTime = endTime
        
    def connection(self):  #method for recovering the data based on the choosen interval
    #limit, url, interval are defined as global variables since they should not be changed since the API limit the request to the server to 1200 per minute
        limit = 1000 
        url = "https://api.binance.com/api/v3/klines"
        interval = '1d'
        year, month, day = map(int, self.startTime.split('-')) #take dates as input from user and split it in 3 variables
        year2, month2, day2 = map(int, self.endTime.split('-'))
        self.startTime=(str(int(dt.datetime(year, month, day).timestamp() *1000)))
        self.endTime= (str(int(dt.datetime(year2,month2,day2).timestamp() *1000)))
       #updating connection parameters 
        self.req_params = {'symbol' : self.symbol, 'interval' : interval, 'startTime' : self.startTime, 'endTime' : self.endTime, 'limit' : limit}  
        # creating pandas dataframe by requesting datas from binance url (defined in url variable) using request library and converting them in a python dictionary with load method from the json library
        DF =pd.DataFrame(json.loads(requests.get(url, params = self.req_params).text))
        #If the request from the server doesn't return any data (that is when the index's lenght is 0) the function connection also doesn't return any value
        if (len(DF.index)== 0): 
         return None 
        DF = DF.iloc[:, 0:6] #else use iloc (integer location based indexing for selection by position, in this case first 6 columns ) with a slice object with intsas input
        DF.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        DF.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in DF.datetime] #setting index with the requested dates 
        DF.head() 
        return(DF)
    
    
class STRATEGY(): 
    def __init__(self,dfneeded):
       self.dfneeded = dfneeded
       
    def stra(self):
        #Creating an empty DataFrame using the index of Bitcoin DataFrame
        g = pd.DataFrame(index=self.dfneeded.index)
        #Creating a column in the empty DataFrame 
        g['Close'] = self.dfneeded['close']
        #Indexing the DataFrame with clean dates (cleaned from the hours)
        g.index = g.index.date
        #Transforming the close prices from an object to a float
        g = g.astype(float)
        g.head()
        #Here we are creating few more column needed by the strategy
        #At first we set the length of the Mooving Average ma=15 perriod
        ma = 15
        #creating the column of the close price, return, moving average and ratio 
        #which shows when the strategy is too far awar from that mean
        g['return'] = np.log(g['Close']).diff()
        g['ma'] = g['Close'].rolling(ma).mean()
        g['ratio'] = g['Close'] / g['ma']
        #Descriptive statistics of ratio column
        g['ratio'].describe()
        #from previpous code we saw that the 'standard' percentile 25% and 75% were too far from the min and max so here we are
        #creating an array to calculate the percentile that we want
        percentiles = [2,5,50,95,98]
        p = np.percentile(g['ratio'].dropna(),percentiles)
        p
        #Strategy plans, definition of when is better go short and long
        #we go short at the 98th percentile
        short = p[-1]
        #we go long at the 2nd percentile
        long = p[0]
        #condition to short
        g['position'] = np.where(g.ratio > short, -1, np.nan)
        #condition to long
        g['position'] = np.where(g.ratio < long, 1, g['position'])
        #according to this strategy we are always in a trade so this command 
        #allows the strategy to invert the position from long to short and from short to long
        g['position'] = g['position'].ffill()
        #dropna for removing missing data before plotting
        g.position.dropna().plot() #dropna for removing missing data before plotting
        #create a coliumn for the strategy return
        g['strat_return'] = g['return'] * g['position'].shift()
        #Comparation trendline of strategy and buy & hold trategy
        plt.plot(np.exp(g['return'].dropna()).cumprod(), label = 'Buy & Hold')
        plt.plot(np.exp(g['strat_return'].dropna()).cumprod(), label = 'Strategy')
        plt.legend()
        #Return of strategy and buy & hold
        print('return of the buy and hold strategy ',np.exp(g['return'].dropna()).cumprod()[-1]-1)
        print('return of the Mean Reverting Strategy ',np.exp(g['strat_return'].dropna()).cumprod()[-1]-1)        
   
    
class CLRM(DATA):
    def __init__(self,x,y):
        self.x = x
        self.y = y
       
    def linreg(self):
        self.x = self.x.to_numpy()
        self.y = self.y.to_numpy()
        self.x = self.x.reshape(-1,1)
        self.y = self.y.reshape(-1,1)
        model = LinearRegression()
        model.fit(self.x,self.y)
        x_test = np.linspace(-0.25,0.25)
        y_pred = model.predict(x_test[:,None])
        plt.scatter(self.x,self.y,5,'g')
        plt.plot(x_test,y_pred,'r')
        plt.legend(['predicted line','observed data'])
        plt.show()
    
    def showstats(self):
        model = sm.OLS(self.x, sm.add_constant(self.y)).fit()
        print(model.summary())
        
        

class tests:
    def __init__(self,x):
        self.x = x       
        
    def adfullertest(self):
        result = adfuller(self.x)
        print (self.x.describe())
        print('')
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
 
    
class Anotherstrategy():
    def __init__(self, dfneeded):
        self.dfneeded= dfneeded
        
    def stra2(self):
        #Creating an empty DataFrame using the index of Bitcoin DataFrame, we call it BTC
        btc = pd.DataFrame(index=self.dfneeded.index)
        #Creating a column in the empty DataFrame 
        btc['price'] = pd.to_numeric(self.dfneeded['close'])
        #Indexing the DataFrame with clean dates (cleaned from the hours)
        btc.index = btc.index.date
        #
        btc['daily_difference'] = btc['price'].diff()
        #Creating signal of the strategy, we can see when we have to go long and when short
        btc['signal'] = np.where(btc['daily_difference']>0,1.0,0.0)
        btc['positions']= btc['signal'].diff()
        #Show the chart
        btc        
        #Creating the signal in the chart
        fig = plt.figure()
        ax1= fig.add_subplot(111,ylabel = "Bitcoin")
        btc['price'].plot(ax = ax1,color = 'b',lw = 2)
        ax1.plot(btc.loc[btc.positions==1.0].index,btc.price[btc.positions==1.0], '^', markersize=7 , color ='g')
        ax1.plot(btc.loc[btc.positions==-1.0].index,btc.price[btc.positions==-1.0], 'v', markersize=7, color ='r')
        #Here we have to see the profit of the strategy
        initial_capital = float(0)
        positions = pd.DataFrame(index=self.dfneeded.index.date).fillna(0.0)
        portfolio = pd.DataFrame(index=self.dfneeded.index.date).fillna(0.0)
        positions['Bitcoin'] = btc['signal']
        portfolio['positions'] = (positions.multiply(btc['price'],axis=0))
        portfolio ['cash'] = initial_capital - (positions.diff().multiply(btc['price'],axis=0)).cumsum()
        portfolio['total']= portfolio['positions']+portfolio ['cash']
        #Show the chart
        #plt.plot(btc['price'],label = 'Price')
        plt.plot(portfolio['total'], color='orange') 
        plt.legend(['Price','Entry','Exit','Strategy'])
        # Show the graph
        plt.show()




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        