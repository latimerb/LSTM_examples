from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import pandas as pd
import pdb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, RepeatVector, TimeDistributed
from keras.layers import LSTM, Dropout
from math import sqrt
from matplotlib import pyplot
from numpy import array
import matplotlib.pyplot as plt

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted
	
def unscale(X, Y, S):
	for i in range(X.shape[0]):
		seg = X[i,:]
		seg = seg.reshape(-1,1)
		us_X = S[i].inverse_transform(seg)
		us_X1d = us_X.reshape(us_X.shape[0],)
			
		Y[i,:] = us_X1d
	return Y

def scalewindow(short_seg):
	short_seg = short_seg.reshape(-1,1)
	scaler = MinMaxScaler(feature_range=(-1,1))
	scaled = scaler.fit_transform(short_seg)
	scaled = scaled.reshape(len(short_seg),)
	return scaled, scaler
	
# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, train.shape[1]-n_lag-n_seq:-n_seq], train[:, -n_seq:]
	print("X.shape: ", X.shape)
	print("Y.shape: ", y.shape)
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	#model.add(Dense(100, kernel_initializer='normal', activation='relu'))
	### EXPERIMENTAL ####
	#model.add(RepeatVector(y.shape[1]))
	#model.add(LSTM(200, activation='relu', return_sequences=True))
	######################
	model.add(Dropout(0.5))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	#for i in range(nb_epoch):
	#	print("epoch ",i, "of ",nb_epoch)
	model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, verbose=1, shuffle=False)
	#	model.reset_states()
	return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, train.shape[1]-n_lag-n_seq:-n_seq], test[i, -n_seq:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()

# load dataset
sub4_aligned = pd.read_csv('../LFP_Prediction/data/train_aligned_sub4_raw.csv',header=None)
sub4_al_vals = sub4_aligned.values



gamma_segs = sub4_al_vals[np.where(sub4_al_vals[:,-1]==1)[0],:]
nongamma_segs = sub4_al_vals[np.where(sub4_al_vals[:,-1]==0)[0],:]

# supervised_dataset is either gamma or nongamma

supervised_dataset = np.zeros((gamma_segs.shape[0],50))
#diff_sup_ds_raw = np.zeros((gamma_segs.shape[0],49))

supervised_dataset = gamma_segs[:,-101:-1]


#for i in np.arange(0,supervised_dataset.shape[0]):
#	diff_sup_ds_raw[0,:] = difference(supervised_dataset[0,:],1)


# scale
scl_sup_ds_raw = np.zeros((supervised_dataset.shape[0],supervised_dataset.shape[1]))

sclr_raw = []

for i in np.arange(0,gamma_segs.shape[0]):
		
	scl_sup_ds_raw[i,:], scl_r = scalewindow(supervised_dataset[i,:])  
		
	sclr_raw.append(scl_r)


# configure
n_lag = 5 # num_lookback
n_seq = 2 # num_predict
n_test = int(0.1*supervised_dataset.shape[0])
N_train = supervised_dataset.shape[0] - n_test
n_epochs = 3
n_batch = 1
n_neurons = 10

# prepare data
#scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
scaler = sclr_raw
train = scl_sup_ds_raw[0:-n_test,:]
test = scl_sup_ds_raw[-n_test:,:]


# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)


# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)



forecasts_arr = np.zeros((len(forecasts),n_seq))
for i in np.arange(0,len(forecasts)):
	forecasts_arr[i,:] = forecasts[i]

# unscale
	
forecasts_us = np.zeros((forecasts_arr.shape[0],forecasts_arr.shape[1]))
forecasts_us = unscale(forecasts_arr, forecasts_us, scaler[-n_test:])

test_us = np.zeros((test.shape[0],test.shape[1]))
test_us = unscale(test,test_us,scaler[-n_test:])


y_pers_test = np.transpose(np.tile(supervised_dataset[N_train:,-n_seq-1], (n_seq,1)))

rmse_pers_test = np.sqrt(np.mean((y_pers_test-test_us[:,-n_seq:])**2,axis=0))
# undifference
# forecasts_us_ud = np.zeros((forecasts_arr.shape[0],forecasts_arr.shape[1]))
# for i in np.arange(0,forecasts_us.shape[0]):
	# forecasts_us_ud[0,:] = inverse_difference()

rmse_lstm_test = np.sqrt(np.mean((test_us[:,-n_seq:]-forecasts_us)**2,axis=0))
print("rmse for testing: ", rmse_lstm_test)


rmse_lstm_test_mean = np.sqrt(np.mean((test_us[:,-n_seq:]-forecasts_us)**2,axis=1))

good_test = np.where(rmse_lstm_test_mean<100)[0]

XX = np.transpose(test_us[good_test])

## FFT
plt.figure()
for i in np.arange(5):
	sp = np.fft.fft(XX[:,i])
	freqs = 1000*np.fft.fftfreq(XX.shape[0])

	plt.subplot(2,1,1)
	plt.plot(XX[:,i])
	plt.subplot(2,1,2)
	plt.plot(freqs[freqs>=0],sp.real[freqs>=0])

plt.show()

# plt.figure()
# plt.hist(rmse_lstm_test_mean)
# plt.show()


################# PLOT ####################

plt.figure()
chunk_size = n_lag + n_seq
for i in np.arange(1,7):

	sample = np.random.randint(0,test.shape[0])
	ax = plt.subplot(2,3,i)
	#ax.set_xticklabels([])
	#ax.set_yticklabels([])
	plt.plot(np.linspace(1,chunk_size,num=chunk_size),supervised_dataset[N_train+sample,-chunk_size:],color='#00FF00')
	plt.plot(np.linspace(n_lag+1,chunk_size,num=n_seq),test_us[sample,-n_seq:],color='#FF0000')
	plt.plot(np.linspace(n_lag+1,chunk_size,num=n_seq),forecasts_us[sample,:],color='orange')
		

fig2 = plt.figure()
plt.subplot(2,1,1)
plt.bar(np.arange(0,n_seq)-0.2,rmse_pers_test*0.1,0.3,label='persistence')
plt.bar(np.arange(0,n_seq)+0.2,rmse_lstm_test*0.1,0.3,label='LSTM')
plt.ylabel('error in uV')
plt.legend()
	
plt.subplot(2,1,2)
plt.bar(np.arange(1,n_seq+1),100*rmse_lstm_test/rmse_pers_test)
plt.plot(np.arange(0,12),100*np.ones((12,)),'r--')
plt.xlim(0,11)





plt.show()
