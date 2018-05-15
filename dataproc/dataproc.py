import matplotlib.pyplot as plt
from dataproc.TimeSeriesDP import TimeSeriesDP
from dataproc import datadb

def getAEData(stock_filename, n_seq, ):
	x, y = datadb.getStock(stock_filename)
	dp=TimeSeriesDP()
	dy = dp.sliding_window(y, n_seq)
	dy = dp.normalize_window_afe(dy)
	return dy, dy