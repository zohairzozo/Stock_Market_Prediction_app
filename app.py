#import libraries
import streamlit as st
import yfinance as yf 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 
import plotly.graph_objects as go
import datetime
from datetime import date, timedelta 
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

#title 
app_name = "<p style='color:red; font-size: 50px; font-weight: bold; '>Forecasting the data</p>"
st.write(app_name,unsafe_allow_html=True)
st.subheader('This is a forecasting app for stock market price for the company of your choice')
#add an image
st.image('https://jupiter.money/blog/wp-content/uploads/2022/08/19.-Investment_strategies_for_beginners-1.jpg')

#take input from user about start and end time
#sidebar 
st.sidebar.header('Select date from below (1-2023 to 1-2024)')

start_date = st.sidebar.date_input('Satrt Date', date(2023,1,1))
end_date = st.sidebar.date_input('End Date', date(2024,1,31))
#add ticker symbol list 
ticker_list = ["AAPL", "MSFT","GOOG","GOOGL","META","TSLA","NVDA","ADBE","PYPL","INTC","CMCSA","CMCSA","NFLX","PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

#fetch data from usre input using yfinance library 

data = yf.download(ticker, start= start_date, end= end_date)
#add Date as column to the dataframe
data.insert(0,"Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', start_date, 'to', end_date)
st.write(data)

#plot the data 
st.write("<p style='color:green; font-size: 50px; font-weight: bold; '>Data Visualization</p> ", unsafe_allow_html=True)
st.write('**Note:** Select your specific data range on the sidebar, or zoom in the plot and select your specefic columns')
fig = px.line(data, x='Date', y=data.columns, title='Play the magic', width=1200, height=600)
st.plotly_chart(fig)

#Selecting from data 
column = st.selectbox('Select colum for forecasting', data.columns[1:]) 
#Subsetting the data 
data = data[['Date', column]]
st.write('Selected data', data)

#ADF check test for stationarity
st.write("<p style='color:green; font-size: 50px; font-style: italic; '>Is data Statioanry?</p> ", unsafe_allow_html=True)
st.write(adfuller(data[column])[1] < 0.05)

# Let's decompose the data 
st.write("<p style='color:green; font-size: 50px; font-weight: bold; '>Decomposition of Data</p> ", unsafe_allow_html=True)
decomposition = seasonal_decompose(data[column], model= 'additive', period = 12)

st.plotly_chart(px.line(x= data["Date"], y= decomposition.trend, title='Trend', width=1500, height=800, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))
st.plotly_chart(px.line(x= data["Date"], y= decomposition.seasonal, title='Seasonality', width=1200, height=600, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Orange'))
st.plotly_chart(px.line(x= data["Date"], y= decomposition.resid, title='Residuals', width=1200, height=600, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))

#Model Building 
st.write("<p style='color:green; font-size: 50px; font-weight: bold; '>Model Parameters</p> ", unsafe_allow_html=True)
p = st.slider('Select the Value of p', 0, 5, 2)
d = st.slider('Select the Value of d', 0, 5, 1)
q = st.slider('Select the Value of q', 0, 5, 2)
seasonal_order = st.number_input('select the value of seasonal P', 0, 24, 12)

model = SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

#print model summary 
st.write("<p style='color:green; font-size: 50px; font-weight: bold; '>Model Sumamry</p> ", unsafe_allow_html=True)
st.write(model.summary())
st.write('---')

# predcit the future values 
st.write("<p style='color:green; font-size: 50px; font-weight: bold; '>Forecasting the data</p> ", unsafe_allow_html=True)
forecast_period = st.number_input("## Eneter forecast Period in days", value= 10 ) 

#predict all valuesfor the forecast period 
predictions = model.get_prediction(start= len(data), end= len(data)+forecast_period)
predictions = predictions.predicted_mean

#add index to results dataframes as dates 
predictions.index = pd.date_range(start= end_date, periods= len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index)
predictions.reset_index(drop=True, inplace=True)
st.write("## Predictions", predictions)
st.write("## Actual Data", data)
st.write("---")

# Plotting predictions
fig = go.Figure()

fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name= 'Actual', line=dict(color='green')))
fig.add_trace(go.Scatter(x=predictions["Date"],y=predictions["predicted_mean"], mode = 'lines', name= 'Predicted', line=dict(color='red')))
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=600)

st.plotly_chart(fig)

# Add button to show plots 
show_plots = False 
if st.button('Show Separate Plots'):
    if not show_plots:
        st.write(px.line(x= data["Date"], y= data[column], title='Actual', width=1200, height=600, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Green'))
        st.write(px.line(x= predictions["Date"], y= predictions["predicted_mean"], title='Predictions', width=1200, height=600, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red'))
        show_plots = True
    else:
        show_plots = False
# Add button to hide plots         
hide_plots = False 
if st.button('Hide Separate Plots'):
    if not hide_plots:  
        hide_plots = True
    else:
        hide_plots = False
        
st.write('---')

st.write('Best Regards')
st.write('Zohair Ahmed Shehzad')
st.write("LinkedIn: https://www.linkedin.com/in/zohair-ahmed2004/ ")
