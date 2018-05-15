from model.seq_rnn_model2 import ProRnnModel,ProRnn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class PredictUpdownModel(ProRnnModel):
	def __init__(self, is_train = False, params=None):
		super(PredictUpdownModel, self).__init__(is_train=is_train, params=params, savename='predict_price')
	
	def __del__(self):
		super(PredictUpdownModel, self).__del__()