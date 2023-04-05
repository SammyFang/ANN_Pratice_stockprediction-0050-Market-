import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

df = pd.read_csv('TWII_history.csv')

data = df.filter(['Close']).values

train_data = data[:5000]
test_data = data[5000:]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_data = scaler.fit_transform(train_data)

def create_dataset(dataset, time_step=1):
    X_data, y_data = [], []
    for i in range(len(dataset)-time_step-1):
        X_data.append(dataset[i:(i+time_step), 0])
        y_data.append(dataset[i+time_step, 0])
    return np.array(X_data), np.array(y_data)

time_step = 60

X_train, y_train = create_dataset(scaled_train_data, time_step)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=64)

scaled_test_data = scaler.transform(test_data)
X_test, y_test = create_dataset(scaled_test_data, time_step)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

future_data = data[-time_step:]
future_data = scaler.transform(future_data)
X_future = future_data.reshape((1, time_step, 1))
for i in range(5):
    prediction = model.predict(X_future)[0][0]
    future_data = np.append(future_data, prediction)
    X_future = future_data[-time_step:]
    X_future = X_future.reshape((1, time_step, 1))

future_data = future_data.reshape((-1, 1))
future_data = scaler.inverse_transform(future_data)
print("Result：")
print(future_data[-5:])