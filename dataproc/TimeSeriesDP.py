import numpy as np
from sklearn.preprocessing import MinMaxScaler
import copy

class TimeSeriesDP:

	def __init__(self, ):
		pass

	# 差分
	def differencing(x, diff_count, period):
		tempx = copy.copy(x)
		for i in range(diff_count):
			tempx[period:] = tempx[period:] - tempx[0:len(tempx) - period]
			tempx = tempx[period:]
		return tempx

	def redifferencing(x, dx, diff_count):
		tempx = copy.copy(x)
		for i in range(diff_count):
			tempx[:] = tempx[:] + dx[:len(tempx)]
		return tempx

	# sliding window x, y
	def windowing(self, x, y, n_seq, dis):
		ns=((len(y)-dis+1) - n_seq)
		dx, dy=[], []
		for i in range(ns):
			dx.append(x[i:i+n_seq])
			dy.append(y[i+n_seq+dis-1])
		return np.array(dx), np.array(dy)

	# sliding window for series x
	def sliding_window(self, x, n_seq):
		ns = (len(x)  + 1 - n_seq)
		dx = []
		for i in range(ns):
			dx.append(x[i:i + n_seq])
		return np.array(dx)

	#  normalize window according first element
	def normalize_window_afe(self, x):
		for i in range(len(x)):
			x0=x[i][0]
			if x0: x[i][:] = (x[i][:]-x0)/x0
			else: x[i][:] = x[i][:]-x0
		return x

	# def renormalize(self, data, origin):
	# 	r=[]
	# 	for i in range(len(data)):
	# 		r.append([origin[i][0][0]*(data[i][0]+1)])
	# 		#print(origin[i][0][0], data[i][0][0])
	# 		#exit()
	# 	if isdifferencing: r=redifferencing(r, TimeSeriesDP.dy[TimeSeriesDP.bound-period:])
	# 	return r
	#
	# def renormalize2(self, data, origin):
	# 	data=TimeSeriesDP.yscaler.inverse_transform(data)
	# 	if isdifferencing: data = redifferencing(data, TimeSeriesDP.dy[TimeSeriesDP.bound-period:])
	# 	return data