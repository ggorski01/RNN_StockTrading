#Kelly Criterion

#   -The Kelly Criterion is a mathematical formula that helps investors and 
# gamblers calculate what percentage of their money they should allocate to each investment or bet.

# -The Kelly Criterion was created by John Kelly, a researcher at Bell 
# Labs, who originally developed the formula to analyze long-distance 
# telephone signal noise.

#   -The percentage the Kelly equation produces represents the size of a 
# position an investor should take, thereby helping with portfolio 
# diversification and money management.

import tensorflow as tf
import numpy as np
import pandas as pd
import yfinance as yf

# Set random seed for reproducibility
tf.random.set_seed(1234)

# Define function to preprocess data
def preprocess_data(data):
    data['log_returns'] = np.log(data['Close']) - np.log(data['Close'].shift(1))
    data = data.dropna()
    x = data['log_returns'].values
    y = np.zeros_like(x)
    y[x > 0] = 1
    return x.reshape(-1, 1), y.reshape(-1, 1)

# Load historical data
symbol = 'SPY'
start_date = '2008-01-01'
end_date = '2023-03-01'
data = yf.download(symbol, start=start_date, end=end_date)

# Preprocess data
x, y = preprocess_data(data)

# Split data into training and validation sets
train_size = int(len(x) * 0.8)
x_train, y_train = x[:train_size], y[:train_size]
x_val, y_val = x[train_size:], y[train_size:]

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(1, 1)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Kelly Criterion function
def kelly_criterion(probability, odds):
    return (probability*odds - (1-probability))/odds

# Train model
history = model.fit(x_train.reshape(-1, 1, 1), y_train, validation_data=(x_val.reshape(-1, 1, 1), y_val), epochs=100)

# Make predictions on validation set
y_pred = model.predict(x_val.reshape(-1, 1, 1))

# Calculate Kelly Criterion for each prediction
kelly_criterion_list = []
for i in range(len(y_pred)):
    kelly_criterion_list.append(kelly_criterion(y_pred[i], 2))

# Calculate total return using Kelly Criterion
total_return = np.prod(kelly_criterion_list) * 100

print('Total return using Kelly Criterion: %.2f%%' % total_return)