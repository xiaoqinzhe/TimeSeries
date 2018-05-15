from model.seq_rnn_model import RnnModel
import tensorflow as tf

class PredictAvgLineModel(RnnModel):
	def __init__(self, is_train = False, params=None):
		super(PredictAvgLineModel, self).__init__(is_train=is_train, params=params, savename='predict_price')
		hide_size=10
		# self.re_outs=tf.reshape(self.cell_outputs, [-1, self.lstm_size*self.n_seq])
		# self.w_o=tf.Variable(tf.truncated_normal([self.lstm_size*self.n_seq, self.output_size], stddev=0.1))
		# self.b_o=tf.Variable(tf.zeros(self.output_size))
		# self.outputs=tf.matmul(self.re_outs, self.w_o)+self.b_o

	def __del__(self):
		super(PredictAvgLineModel, self).__del__()