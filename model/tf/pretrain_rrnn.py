import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
import numpy as np

class PretrainRRnn:
	def __init__(self, is_train, params):
		self.is_train=is_train
		self.batch_size=params['batch_size']
		self.n_seqs=params['n_seqs']
		self.lstm_size=params['lstm_size']
		self.num_layers=params['num_layers']
		self.keep_prob=params['keep_prob']
		self.learning_rate=params['learning_rate']
		self.input_size=params['input_size']
		self.output_size=params['output_size']
		self.num_scopes=len(self.n_seqs[0])
		self.grad_clip=2
		self.targets=tf.placeholder(tf.float32, shape=[None, self.output_size])
		#lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
		#drop=tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
		self.inputs=[]
		self.initial_states=[]
		self.states=[]
		self.in_outputs=[]
		self.scope_scales=[]
		self.pretrain_outputs=[]
		self.pretrain_losses = []
		self.pretrain_optimizers = []
		for i in range(self.num_scopes):
			with tf.variable_scope("lstm_scope"+str(i)):
				self.inputs.append(tf.placeholder(tf.float32, shape=[None, self.n_seqs[1][i], self.input_size]))
				if i>0: self.scope_scales.append(tf.placeholder(tf.float32, [2], "scale"))
				cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)])
				#self.initial_states.append(cell.zero_state(self.batch_size, tf.float32))
				self.initial_states.append(cell.zero_state(self.batch_size, tf.float32))
				cell_inputs=self.inputs[i]
				cell_outputs, state=tf.nn.dynamic_rnn(cell, cell_inputs,initial_state=self.initial_states[i], dtype=tf.float32)
				self.states.append(state)
				re_outs=cell_outputs[:, -1, :]
				self.add_extra_optimizer(re_outs)
				if i>0:
					w=tf.get_variable("rw", shape=(self.lstm_size*2, self.lstm_size))
					b=tf.get_variable("rb", shape=(self.lstm_size))
					# re_outs=tf.matmul(tf.concat([re_outs, self.in_outputs[i-1]], axis=1), w)+b
					re_outs=tf.nn.sigmoid(tf.matmul(tf.concat([re_outs, self.in_outputs[i-1]], axis=1), w)+b)
					# w = tf.get_variable("rw2", shape=(self.lstm_size , self.lstm_size))
					# b = tf.get_variable("rb2", shape=(self.lstm_size))
					# re_outs = tf.nn.sigmoid(tf.matmul(re_outs, w) + b)
					re_outs=tf.add(self.scope_scales[i-1][1]*re_outs, self.scope_scales[i-1][0]*self.in_outputs[i-1])
				self.in_outputs.append(re_outs)
		w_o=tf.get_variable('weight', [self.lstm_size, self.output_size])
		b_o=tf.get_variable('bias', [self.output_size])
		self.outputs=tf.matmul(self.in_outputs[-1], w_o)+b_o
		#loss and optimizer
		#  self.loss=tf.sqrt(tf.reduce_mean(tf.square(self.targets-self.outputs)))
		self.loss = tf.reduce_mean(tf.square(self.targets - self.outputs))
		# self.loss=tf.losses.huber_loss()
		if is_train:
			#clipping gradients optimizer
			self.tvars=tf.trainable_variables()
			self.grads, _=tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars), self.grad_clip)
			# self.lr=tf.train.
			self.optimizer=tf.train.AdamOptimizer(self.learning_rate)
			self.bp_operation = self.optimizer.minimize(self.loss)
			#self.bp_operation = self.optimizer.apply_gradients(zip(self.grads, self.tvars))

	def lstm_cell(self):
		lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
		if not self.is_train or self.keep_prob >= 1: return lstm
		drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
		return drop

	def add_extra_optimizer(self, out):
		with tf.variable_scope('optimizer'):
			w_o = tf.get_variable('weight', [self.lstm_size, self.output_size])
			b_o = tf.get_variable('bias', [self.output_size])
			outputs = tf.matmul(out, w_o) + b_o
			self.pretrain_outputs.append(outputs)
			# loss and optimizer
			self.pretrain_losses.append(tf.sqrt(tf.reduce_mean(tf.square(self.targets - outputs))))
			self.pretrain_optimizers.append(tf.train.AdamOptimizer(self.learning_rate).minimize(self.pretrain_losses[-1]))