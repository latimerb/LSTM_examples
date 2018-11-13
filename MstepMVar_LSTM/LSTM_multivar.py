# multivariate multi-step encoder-decoder lstm

#https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

from math import sqrt
from numpy import split
import numpy as np
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import set_random_seed

set_random_seed(27365)

# split a univariate dataset into train/test sets
def split_dataset(data, n_out):
	# split into standard weeks
	n_segs = np.floor((len(data))/n_out)
	n_train = np.floor(0.8 * n_segs)
	n_train = int(np.floor(n_train/n_out)*n_out) # make sure it's a factor of n_out
	
	print("n_segs: ", n_segs)
	print("n_train: ", n_train)
	
	
	train, test = data[0:-n_train], data[-n_train:]
	
	print("train.shape: ", train.shape)
	# restructure into windows of weekly data
	train = array(split(train, len(train)/n_out))
	test = array(split(test, len(test)/n_out))
	return train, test

# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	print("actual.shape[1]: ", actual.shape[1])
	for i in range(actual.shape[1]):
		# calculate mse
		
		print("!! ", i)
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
def to_supervised(train, n_input, n_out):
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)

# train the model
def build_model(train, n_input, n_out):
	# prepare data
	train_x, train_y = to_supervised(train, n_input, n_out)
	# define parameters
	verbose, epochs, batch_size = 1, 4, 100
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(LSTM(100, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(50, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into [1, n_input, n]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

# evaluate a single model
def evaluate_model(train, test, n_input, n_out):
	# fit model
	model = build_model(train, n_input, n_out)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores, predictions



# load the new file
#dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#dataset = dataset.values

dataset = read_csv('ex_LFP_5chan.csv')
dataset = dataset.values[0:20000,0:5] # length of dataset must be divisible by n_out

#scaler = MinMaxScaler(feature_range=(-2,2))
#scaled = scaler.fit_transform(short_seg)


n_channels = dataset.shape[1]

n_input = 30 #num_lookback
n_out = 10 #num_predict


# split into train and test
train, test = split_dataset(dataset,n_out)

# make variables for scaled data
train_scl = np.zeros((train.shape[0],train.shape[1],train.shape[2]))
test_scl = np.zeros((test.shape[0],test.shape[1],test.shape[2]))

train_us = train
test_us = test

sclr_train = []
sclr_test = []

for i in np.arange(train.shape[0]):
	scl1 = MinMaxScaler(feature_range=(-1,1))
	scaled1 = scl1.fit_transform(train[i,:,0:1])
	train_scl[i,:,0:1] = scaled1
	scl2 = MinMaxScaler(feature_range=(-1,1))
	scaled2 = scl2.fit_transform(train[i,:,1:2])
	train_scl[i,:,1:2] = scaled2
	scl3 = MinMaxScaler(feature_range=(-1,1))
	scaled3 = scl3.fit_transform(train[i,:,2:3])
	train_scl[i,:,2:3] = scaled3
	scl4 = MinMaxScaler(feature_range=(-1,1))
	scaled4 = scl4.fit_transform(train[i,:,3:4])
	train_scl[i,:,3:4] = scaled4
	scl5 = MinMaxScaler(feature_range=(-1,1))
	scaled5 = scl5.fit_transform(train[i,:,4:5])
	train_scl[i,:,4:5] = scaled5
	# only append scaler for first variable
	sclr_train.append(scl1)

for i in np.arange(test.shape[0]):
	scl1 = MinMaxScaler(feature_range=(-1,1))
	scaled1 = scl1.fit_transform(test[i,:,0:1])
	test_scl[i,:,0:1] = scaled1
	scl2 = MinMaxScaler(feature_range=(-1,1))
	scaled2 = scl2.fit_transform(test[i,:,1:2])
	test_scl[i,:,1:2] = scaled2
	scl3 = MinMaxScaler(feature_range=(-1,1))
	scaled3 = scl3.fit_transform(test[i,:,2:3])
	test_scl[i,:,2:3] = scaled3
	scl4 = MinMaxScaler(feature_range=(-1,1))
	scaled4 = scl4.fit_transform(test[i,:,3:4])
	test_scl[i,:,3:4] = scaled4
	scl5 = MinMaxScaler(feature_range=(-1,1))
	scaled5 = scl5.fit_transform(test[i,:,4:5])
	test_scl[i,:,4:5] = scaled5
	# only append scaler for first variable
	sclr_test.append(scl1)	



# train is now N_train (time chunks) x 7 (time samples) x 8 (channels)

train_scl = train_scl.reshape(train_scl.shape[0],train_scl.shape[1],train_scl.shape[2])
test_scl = test_scl.reshape(test_scl.shape[0],test_scl.shape[1],test_scl.shape[2])

train_us = train_us.reshape(train_us.shape[0],train_us.shape[1],train_us.shape[2])
test_us = test_us.reshape(test_us.shape[0],test_us.shape[1],test_us.shape[2])

# evaluate model and get scores

score, scores, preds = evaluate_model(train_scl, test_scl, n_input, n_out)


# unscale the preds
preds = preds.reshape(preds.shape[0],preds.shape[1])
preds_us = np.zeros((preds.shape[0],preds.shape[1]))
for i in np.arange(preds.shape[0]):
	seg = preds[i,:]
	seg = seg.reshape(-1,1)
	inv_scale = sclr_test[i].inverse_transform(seg)
	preds_us[i,:] = inv_scale[:,0]


# PERSISTENCE FORECAST
y_pers_test = np.zeros((test_us.shape[0],test_us.shape[1]))

# first test is from the last sample of train
y_pers_test[0,:] = np.transpose(np.tile(train_us[-1,-1,0], (n_out,1)))
for i in np.arange(1,test.shape[0]):
	y_pers_test[i,:] = np.transpose(np.tile(test_us[i-1,-1,0], (n_out,1)))


plt.figure()
for i in np.arange(6):
	ax = plt.subplot(2,3,i+1)
	sample = np.random.randint(0,test_us.shape[0])
	plt.plot(np.arange(1,3*n_out+1),np.concatenate((test_us[sample-1,:,0],test_us[sample,:,0],test_us[sample+1,:,0])),color='#00FF00')
	plt.plot(np.arange(2*n_out+1,3*n_out+1),test_us[sample+1,:,0],color='#FF0000')
	#plt.plot(np.arange(n_out+1,2*n_out+1),y_pers_test[sample+1,:],color='orange')
	plt.plot(np.arange(2*n_out+1,3*n_out+1),preds_us[sample+1,:],color='orange')
	ax.set_xticklabels([])
	ax.set_yticklabels([])
plt.tight_layout()

summarize_scores('lstm', score, scores)

rmse_lstm = np.sqrt(np.mean((preds_us - test_us[:,:,0])**2,axis=0))

rmse_pers = np.sqrt(np.mean((y_pers_test[:,:] - test_us[:,:,0])**2,axis=0))

print("LSTM RMSE: ", rmse_lstm)

print("PERS RMSE: ",rmse_pers)


fig2 = plt.figure()
plt.subplot(2,1,1)
plt.bar(np.arange(0,n_out)-0.2,rmse_pers*0.1,0.3,label='persistence')
plt.bar(np.arange(0,n_out)+0.2,rmse_lstm*0.1,0.3,label='LSTM')
plt.ylabel('error in uV')
plt.legend()
	
plt.subplot(2,1,2)
plt.bar(np.arange(1,n_out+1),100*rmse_lstm/rmse_pers)
plt.plot(np.arange(0,12),100*np.ones((12,)),'r--')
plt.xlim(0,11)

plt.show()
# # summarize scores
# summarize_scores('lstm', score, scores)

# # plot scores
# days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
# pyplot.plot(days, scores, marker='o', label='lstm')
# pyplot.show()