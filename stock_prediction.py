#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from keras.models import load_model
from nsepy import get_history
from datetime import date
import datetime
import streamlit as st

from PIL import Image




st.title("Developed by ***Virander Kumar***")


image = Image.open('nse.jpg')
st.image(image,use_column_width=True)

st.title("NSE Real-Time Stocks Analysis and Predictions")

st.header("Select the stock and check its next day predicted value")

st.subheader("This study is mainly confined to the s tock market behavior and is \
            intended to devise certain techniques for investors to make reasonable\
            returns from investments .")

st.subheader("Though there were a number of studies , which\
            deal with analysis of stock price behaviours , the use of control chart\
            techniques and fai lure time analysis would be new to the investors. The\
            concept of stock price elast icity,\
            introduced in this study, will be a good\
            tool to measure the sensitivity of stock price movements.")

st.subheader("In this study, \
             Predictions for the close price is suggested for the National Stock Exchange index,\
            Nifty,\
            based on Long Short Term Based (LSTM)\
            method.") 

st.subheader("We make predictions based on the last 30 days Closing price data\
            which we fetch from NSE India website in realtime.")

#Create a slidebar header
st.sidebar.header('User Input')

#Creating user inp in the side bar
#start_date = st.sidebar.text_input("Start Date", "2000,01,01")
#end_date = st.sidebar.text_input("End Date", "2020,10,14")

start_date = st.sidebar.date_input("Start Date", date(2000,1,1))
end_date = st.sidebar.date_input("End Date", date.today())
stock_symbol = st.sidebar.text_input("Stock Symbol","INFY")

#to get the historical data
##df = web.DataReader(com, 'yahoo', st_date, end_date)  # Collects data
##df.reset_index(inplace=True)
##df.set_index("Date", inplace=True)

# get abfrl real time stock price
#df1 = get_history(stock_symbol, start_date, end_date)
#df1 = get_history(symbol=stock_symbol, start=start_date, end=end_date)
#df1 = get_history(symbol=stock_symbol, start=date(2010,1,1), end=date.today())
df1 = get_history(symbol=stock_symbol, start=start_date, end=end_date)
df1['Date'] = df1.index

st.header(stock_symbol.upper()+" NSE DataFrame:")
 
# Insert Check-Box to show the snippet of the data.
if st.checkbox('Show Raw Data'):
    st.subheader("Showing raw data---->>>")	
    st.dataframe(df1)





## Predictions and adding it to Dashboard
#Create a new dataframe
new_df = df1.filter(['Close'])
#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(new_df)
#Get teh last 30 day closing price 
last_30_days = new_df[-30:].values
#Scale the data to be values between 0 and 1
last_30_days_scaled = scaler.transform(last_30_days)
#Create an empty list
X_test = []
#Append teh past 1 days
X_test.append(last_30_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#load the model
model = load_model("abfrl.model")
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling 
pred_price = scaler.inverse_transform(pred_price)

# next day
#NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)
NextDay_Date = end_date + datetime.timedelta(days=1)

st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
st.markdown(pred_price)

##visualizations
st.subheader("Close Price VS Date Interactive chart for analysis:")
st.area_chart(df1['Close'])

st.subheader("Line chart of Open and Close for analysis:")
st.area_chart(df1[['Open','Close']])

st.subheader("Line chart of High and Low for analysis:")
st.line_chart(df1[['High','Low']])


