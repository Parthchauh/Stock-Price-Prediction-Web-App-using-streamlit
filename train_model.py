import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

# Load your dataset
data_file_path = "Data.csv"  # Update this path
df = pd.read_csv(data_file_path)

# Data preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Use only the 'Close' column for training
data = df['Close'].values
data = data.reshape(-1, 1)

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Splitting data into training and testing sets
train_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:train_size]

# Reducing the dataset size (optional)
train_data = train_data[:2000]  # Use only the first 2000 data points

# Preparing the training data with a smaller window size
x_train = []
y_train = []

for i in range(50, len(train_data)):
    x_train.append(train_data[i-50:i])
    y_train.append(train_data[i])

x_train, y_train = np.array(x_train), np.array(y_train)

# Building the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Setting up early stopping
early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1)

# Training the model with a larger batch size
model.fit(x_train, y_train, batch_size=64, epochs=5, callbacks=[early_stopping])

# Saving the model
model.save("Latest_stock_price_model.keras")
