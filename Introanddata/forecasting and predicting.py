import pandas as pd
import quandl, math, datetime, os
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# We need to figure out what features of this data set to use for our stock price prediction.
# We'll choose only somewhat relevant features, in this case the Adjusted open, Adjusted close,
# The adjusted high and low stock price, the Adjusted close and the Adjusted volume. 
# Note that 'Adjusted' simply means the price of the stock after such things as  stock splits. 
# We could use the non-adjusted features as well, but using both adjusted and non-adjusted is 
# not useful as they both essentially describe the same things. 
quandl.ApiConfig.api_key = os.environ['QUANDL_API_KEY']
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

# Here, we're going to create two new features computed from the features already given 
# in the data set.
# We're going to need these features to track volatility in the stock
# which is an essential feature for predicting a new stock price. 

#The first features is the high low percentage, calculated by subtracting the
# adjusted high price from the close and dividing by the close multiplied by 100.0 (to get a proper percentage value)
#
df['HL_PCT'] =  (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

# Our next features is the Percentage price change obtained by subtracting the adjusted close
# from the adjusted open and dividing  by the adjusted open multiplied by 100.0
df['PCT_change'] =  (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

# Once we have the features, we want to then get the labels, i.e. for a classifier, we want to know
# whether or not we have a True or a False value. 

# Here we're using the Adjusted close as our label name.  
forecast_col = 'Adj. Close'

# With machine learning, we will replace any NaN values with a real number (even though it's an outlier). 
# This is a better choice than getting rid of columns that don't have all the data. 
df.fillna(-99999, inplace=True)

# Now we'll define our regression algorithm
# Let's first set how many days back we want to go  as a percentage of the number of days in the data frame. 
# Here we'll set it to go back one percent of the total number of days in the data frame. 
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

# Here we use the pandas shift method to shift the forecast label values out by the amount
# specified in the forecast_out variable.  Basically, the label will contain the adjusted close
# value 34 days (as specified by the forecast_out variable) into the future. 
df['label'] = df[forecast_col].shift(-forecast_out)

# Create an array from df, with no label column.
X = np.array(df.drop(['label'],1))

# This is suppose to scale the values, and create a clamp. Speeds up training the data.
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out:]


df.dropna(inplace=True)

y = np.array(df['label'])


# Now, let's get two sets of values.  
# Set 1.  The training data for our linear regression algorithm.
# Set 2.  The testing data to see how well our algorithm learned from the training data. 

# Here we're going to split up the data we got from Quandl for the training feature data (X_train) and the training label data (y_train)
# and the testing feature data (X_test) and the testing label data (y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Let's choose the SKLearn's Linear Regression algorithnm for our ML example
clf = LinearRegression(n_jobs=-1)

# Let's run the training data through the algorithm. 
clf.fit(X_train, y_train)

# Now let's see how well the training set compares to the testing data.
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# Iterating through forecast_set, taking each forecast and day, and setting those as the values in the data frame. Basically making future features NaN or not a number
for i in forecast_set:
    # The index of the data inputs are the dates, this just updates those into the future, sense we don't have them, but they can be predicted.
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # This takes the first through second to last column, sets it to NaN. This is the range(len(df.columns)-1) The last column is set to i, aka forecast.
    # We do this because the first through second to last columns, (like adj. Close, adj. open, etc..) we don't have any actual data for, so we set these columns to NaN
    # Then we set the last column (forecast) to i, which is just data in the list forecast_set. forecast_set being our predicated prices)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
