from model.seq_rnn_model2 import RnnModel
import tensorflow as tf

class PredictPriceModel(RnnModel):
	def __init__(self, is_train = False, params=None, savename='lstm'):
		super(PredictPriceModel, self).__init__(is_train=is_train, params=params, savename=savename)

	def __del__(self):
		super(PredictPriceModel, self).__del__()