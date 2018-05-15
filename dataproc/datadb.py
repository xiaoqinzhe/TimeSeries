import numpy as np
import pandas as pd
import netCDF4 as cdf
import copy
import matplotlib.pyplot as plt

stockPath="./data/stocks/"
stockFilename=[ "000002.csv",
				"000001.csv",
				"399300.csv",
				"600000.csv",
				"600519.csv",
				"002120.csv"
				]
smoothLengths={'sunspot' : 13,
			   'temperature' : 4}

def selectTime(df, time_index, start_time, end_time):
	pass

def smoothSeries(df, sm):
	a = np.zeros(len(df) - sm + 1, dtype=np.float32)
	for i in range(len(a)):
		a[i] = np.mean(df[i:i + sm])
	return a

def getStock(filename, price_index=4, start_time=None, end_time=None):
	df=pd.read_csv(stockPath+filename)
	if start_time: df=selectTime(df, start_time, end_time)
	df = df.values[:,price_index]
	return df[:np.newaxis], df[:,np.newaxis]

def getSunspotSeq(params=None):
	path = './data/sunspot/monthly-sunspot-number-zurich-17.csv'
	df = pd.read_csv(path)
	df = df.values[:, 1]
	sm = 13
	a = smoothSeries(df, sm)
	return a[:, np.newaxis], copy.copy(a[:, np.newaxis])

def getTemperature(params=None):
	path='./data/temperature/daily-minimum-temperatures-in-me.csv'
	df=pd.read_csv(path)
	sm=4
	a=smoothSeries(df, sm)
	return a[:, np.newaxis], copy.copy(a[:, np.newaxis])

def getAir(params=None):
	path = '../data/air/air.mon.anom.nc'
	df = cdf.Dataset(path, "a")
	print(df.dimensions)
	return df[:, np.newaxis], df[:, np.newaxis]

def getHouseConsumption(params=None):
	path = '../data/house_consumption/household_power_consumption.txt'
	df = pd.read_csv(path,delimiter=';', na_values='?')
	datasize=16000
	print(df.columns)
	print(df.dtypes)
	# print(df.head())
	# print(df.describe())
	plt.plot(df.loc[:datasize,'Global_active_power']*1000/60 - df.loc[:datasize,'Sub_metering_1'] - df.loc[:datasize,'Sub_metering_2'] - df.loc[:datasize,'Sub_metering_3'])
	plt.show()
	df=df.values
	return df[:, np.newaxis], df[:, np.newaxis]

if(__name__=='__main__'):
	x, y = getStock(stockPath+stockFilename[0])
	import matplotlib.pyplot as plt
	print(np.mean(y), np.var(y))
	plt.plot(np.squeeze(y))
	plt.show()