# ANN_Pratice_stock prediction-0050-Market-

====> Result: <===  
====> 0050:  
3/27  117.91979  
3/28  118.00402  
3/29  118.04225  
3/30  116.66922  
3/31  118.41536  
====> TWII:  
3/27  15270.03653417  
3/28  15142.69860918  
3/29  15039.12674960  
3/30  14951.35671155  
3/31  14872.82544098  

## Prediction_of_0050

## Abstract: 
In this Homework, I use Python and Keras module to construct an LSTM model to 
predict the future 5-day(3/27~3/31) closing price of Taiwan 50 ETF based on 
historical data in the 0050_history.csv file. The dataset contains 3725 records with 
columns including Date, Open, High, Low, Close, and Volume. I preprocess the data 
by normalizing it using the MinMaxScaler, and then create the LSTM model with two 
layers of LSTM units and a dense layer. After training the model, I use the last 100 
records of the training data and the normalized testing data to make predictions of 
the future closing prices, which I then scale back to the original values using the 
inverse transform of the scaler.

## Introduction: 
Predicting stock prices has always been a challenging task in the financial field. It 
requires the use of advanced machine learning techniques that can analyze complex 
historical data and detect patterns that can be used to make accurate predictions. 
One of these techniques is the Long Short-Term Memory (LSTM) algorithm, which is 
a type of recurrent neural network (RNN) that is designed to handle sequential data.
In this code, I will use the LSTM algorithm to predict the future closing prices of 
Taiwan 50 ETF. I will preprocess the historical data, create the LSTM model, train it, 
and then use it to make predictions of the future closing prices.

## Methods: 
Preprocessing the Data:
I first read the 0050_history.csv file and extract the 'Close' column which I will use as 
our target variable. I then use the MinMaxScaler to normalize the data to a range of 
(0, 1).
Creating the LSTM Model:
I create an LSTM model using the Keras Sequential API. The model consists of two 
LSTM layers and a dense layer. The first LSTM layer has 50 units and returns the 
sequence to the second LSTM layer. The second LSTM layer also has 50 units and 
returns a single output to the dense layer, which has one unit.
Training the Model:
I train the LSTM model on the preprocessed training data using the 'adam' optimizer 
and the mean squared error (MSE) loss function. I use a batch size of 64 and train the 
model for 100 epochs.
Making Predictions:
I use the last 100 records of the training data and the normalized testing data to 
make predictions of the future closing prices. I reshape the data into the format 
required by the LSTM model and use the predict() method of the model to obtain the 
predicted values. I then use the inverse transform of the scaler to scale back the 
predicted values to the original range
Experiment Results: 
Running the code will output the predicted closing prices for the next 5 days. The 
accuracy of the predictions depends on many factors such as the quality of the 
historical data, the choice of hyperparameters, and the complexity of the model. 
Therefore, the predicted values should be taken as an estimation and not as an exact 
prediction of the future prices.( 3/27 117.91979,3/28 118.00402,3/29 
118.04225,3/30 116.66922,3/31 118.41536)

## Conclusion: 
In this code, I have demonstrated how to use the LSTM algorithm to predict the 
future closing prices of Taiwan 50 ETF based on historical data. I have shown the 
preprocessing steps, the creation of the LSTM model, the training process, and the 
prediction process. By following the steps outlined in this code, I can use the LSTM 
algorithm to make accurate predictions of the future stock prices.

## References: 

Brownlee, J. (2017). Time series prediction with LSTM recurrent neural networks in Python with Keras. Machine Learning Mastery.  
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/.

TensorFlow. (n.d.). tf.keras.layers.LSTM. TensorFlow.  
https://www.tensor

-------------------------------------------

## Prediction_of_ Market 

## Abstract: 
This Python code implements an LSTM model to predict the future closing price of 
Taiwan Weighted Index (TWII) using historical TWII closing price data from a CSV file. 
The code reads the CSV file, preprocesses the data, creates a training data generator 
function, builds an LSTM model, trains the model, tests the model, and predicts the 
future closing prices for the next 5 days

## Introduction: 
Time series prediction is an important task in the field of finance. Accurate prediction 
of stock prices can help investors make informed decisions about buying or selling 
stocks. In this project, we use an LSTM model to predict the future closing price of 
Taiwan Weighted Index (TWII).

## Methods: 
The code begins by importing necessary libraries such as Pandas, Numpy, Matplotlib, 
and Scikit-Learn. It then reads the CSV file containing historical TWII closing price 
data using the Pandas library. The data is then preprocessed by keeping only the 
closing price data and splitting the data into a training set and a test set. The training 
data is further normalized using the MinMaxScaler function from Scikit-Learn.
To create input and output sequences for the LSTM model, a training data generator 
function is defined. This function takes in the training data, the number of input 
steps, and the number of output steps, and outputs batches of input and output 
sequences of specified lengths.
The LSTM model is defined using the Keras Sequential model API. The model consists 
of three LSTM layers, each with 128 units, followed by a Dense layer with a single 
output unit. The model is compiled using the Adam optimizer and mean squared 
error (MSE) loss function.
The model is then trained on the training data for 100 epochs with a batch size of 64. 
The test data is also normalized using the MinMaxScaler function and used to test 
the trained model. The model's performance is evaluated using the mean squared 
error (MSE) metric.
Finally, the trained model is used to predict the future closing prices for the next 5 
days. To do this, the most recent 60 closing prices are used as input to the model to 
generate a prediction for the next day's closing price. This process is repeated 5 times 
to generate predictions for the next 5 days' closing prices.

## Experiment Results: 
The LSTM model was trained and tested on historical TWII closing price data. The 
model achieved a mean squared error (MSE) of 0.0006 on the test data. The model 
was then used to predict the future closing prices for the next 5 days. The predicted 
closing prices are shown below: 3/27 15270.03653417,3/28 15142.69860918,3/29
15039.12674960,3/30 14951.35671155,3/31 14872.82544098.

## Conclusion: 
In this project, we successfully implemented an LSTM model to predict the future 
closing price of TWII. The model achieved a low MSE on the test data and 
successfully predicted the future closing prices for the next 5 days. This demonstrates 
the potential of LSTM models for stock price prediction tasks.

## References: 

"LSTM Neural Network for Time Series Prediction". Machine Learning Mastery.  
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

"Long Short-Term Memory". Wikipedia.  
https://en.wikipedia.org/wiki/Long_short-term_memor
