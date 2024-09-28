import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Set up Streamlit app title
st.title("Stock Price Predictor App")

# User input for stock symbol
stock = st.text_input("Enter the Stock ID", "TCS")

# Load your dataset
# Change the path to where your data file is located
data_file_path = "Data.csv"  # Update this path
df = pd.read_csv(data_file_path)

# Filter data for the selected stock symbol
df_stock = df[df['Symbol'] == stock].copy()

# Convert 'Date' to datetime format and sort by date
df_stock['Date'] = pd.to_datetime(df_stock['Date'])
df_stock.sort_values('Date', inplace=True)

# Display stock data
st.subheader("Stock Data")
st.write(df_stock)

# Set the Date as the index
df_stock.set_index('Date', inplace=True)

# Splitting the data for testing
splitting_len = int(len(df_stock) * 0.7)
x_test = pd.DataFrame(df_stock['Close'][splitting_len:])

# Define a plotting function
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data['Close'], 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Plot moving averages
for days in [250, 200, 100]:
    ma_column = f'MA_for_{days}_days'
    df_stock[ma_column] = df_stock['Close'].rolling(days).mean()
    st.subheader(f'Original Close Price and MA for {days} days')
    st.pyplot(plot_graph((15, 6), df_stock[ma_column], df_stock, 0))

# Plot both 100-day and 250-day moving averages
st.subheader('Original Close Price MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), df_stock['MA_for_100_days'], df_stock, 1, df_stock['MA_for_250_days']))

# Scale the data for prediction
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

# Prepare the data for prediction
x_data = []
y_data = []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

# Convert to numpy arrays
x_data, y_data = np.array(x_data), np.array(y_data)

# Load the trained model
model = load_model("Latest_stock_price_model.keras")

# Make predictions using the loaded model
predictions = model.predict(x_data)

# Inverse transform the predictions
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare plotting data
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=df_stock.index[splitting_len + 100:]
)

# Display original vs predicted values
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

# Plot the original close price vs predicted close price
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([df_stock['Close'][:splitting_len + 100], ploting_data], axis=0))
plt.legend(["Data - not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)
