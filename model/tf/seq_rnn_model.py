import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataproc.datadb import *
from datetime import datetime

default_params={1:{'batch_size':50, 'n_seq':20, 'lstm_size':20, 'num_layers':3, 'keep_prob':1, 
                    'learning_rate':0.01, 'input_size':1, 'output_size':1, 'num_epochs':100},
                2:{'batch_size':50, 'n_seq':20, 'lstm_size':20, 'num_layers':1, 'keep_prob':1, 
                    'learning_rate':0.01, 'input_size':5, 'output_size':1, 'num_epochs':100}}

class RnnModel:
    def __init__(self, is_train = False, params=None, savename='default'):
        self.is_train=is_train
        self.save_path='model/rnnmodels/'+'default'
        #self.data_size=20000
        if params==None: params=default_params[1]
        self.batch_size=params['batch_size']
        self.n_seq=params['n_seq']
        self.lstm_size=params['lstm_size']
        self.num_layers=params['num_layers']
        self.keep_prob=params['keep_prob']
        self.learning_rate=params['learning_rate']
        self.input_size=params['input_size']
        self.output_size=params['output_size']
        self.grad_clip=8
        self.trainx=None
        self.trainy_=None
        self.trainy=None
        self.num_epochs=params['num_epochs']
        self.initRnn()

    def __del__(self):
        self.sess.close()

    def initRnn(self):
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        self.sess=tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        #self.sess=tf.Session()
        self.inputs=tf.placeholder(tf.float32, shape=[None, self.n_seq, self.input_size])
        self.targets=tf.placeholder(tf.float32, shape=[None, self.output_size])

        #lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        #drop=tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        self.cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)])
        self.initial_state=self.cell.zero_state(self.batch_size, tf.float32)
        self.cell_inputs=self.inputs
        self.cell_outputs, self.state=tf.nn.dynamic_rnn(self.cell, self.cell_inputs, dtype=tf.float32)
        print(self.initial_state[0].c)
        self.re_outs=tf.reshape(self.cell_outputs, [-1, self.lstm_size*self.n_seq])
        w_o=tf.Variable(tf.truncated_normal([self.lstm_size*self.n_seq, self.output_size], stddev=0.1))
        b_o=tf.Variable(tf.zeros(self.output_size))
        self.outputs=tf.matmul(self.re_outs, w_o)+b_o

        #loss and optimizer
        self.loss=tf.sqrt(tf.reduce_mean(tf.square(self.targets-self.outputs)))
        #clipping gradients optimizer
        self.tvars=tf.trainable_variables()
        self.grads, _=tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars), self.grad_clip)
        self.optimizer=tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.grads, self.tvars))
        #optimizer=tf.train.AdamOptimizer(0.1).minimize(loss)

        #predict
        #self.meandiff=tf.reduce_mean(self.targets-self.outputs)

        self.saver=tf.train.Saver()
        if not self.is_train: self.saver.restore(self.sess, self.save_path)
    
    def get_batches(self, dx, dy):
        data=[]
        for i in range(len(dx)):
            data.append([dx[i], dy[i]])
        np.random.shuffle(data)
        nb=len(data)//self.batch_size
        for i in range(nb):
            x, y = [], []   
            for j in range(i*self.batch_size, (i+1)*self.batch_size):
                x.append(data[j][0])
                y.append(data[j][1])
            yield x, y

    def lstm_cell(self):
        lstm=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
        if self.keep_prob>=1: return lstm
        drop=tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
        return drop

    def train(self, x_train, y_train, validation=None):
        #with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
        self.sess.run(tf.global_variables_initializer())
        new_state=self.sess.run(self.initial_state)
        i=0
        for e in range(self.num_epochs):
            batches=self.get_batches(x_train, y_train)
            losssum=i=0
            for b in batches:
                i+=1
                feed={self.inputs:b[0], self.targets:b[1]}
                #feed={self.inputs:b[0], self.targets:b[1]}
                #print(b[0][-1][-5:], b[1][-1])
                lo, new_state, _ = self.sess.run([self.loss, self.state, self.optimizer], feed_dict=feed)
                losssum+=lo
                #print(new_state)
                #print(e,i,lo)
                #if i%20==0: 
                #    print(e,i,lo)
            testloss=0
            if validation: testloss=self.test(validation[0], validation[1])
            print(e,'epoch', losssum/i, testloss)
        self.saver.save(self.sess, self.save_path)

    def predictV(self, seq):
        #print([[seq,[0]]])
        pre=self.sess.run(self.outputs, feed_dict={self.inputs: seq})
        return pre

    def test(self, x_test, y_test):
        #test
        #with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
        lo, outputs=self.sess.run([self.loss, self.outputs], feed_dict={self.inputs:x_test, self.targets:y_test})
        #print('loss', lo)
        return lo
        #原始
        for i in range(self.output_size):
            plt.figure()
            #print(len(origin), len(outputs))
            plt.title('output index: '+str(i))
            plt.plot(outputs[:,i].reshape(-1), label='predict')
            plt.plot(np.array(y_test)[:,i].reshape(-1), label='goal')
            plt.legend(loc=0)
        #plt.show()
        return outputs
        #print(len(test_data[1]), len(output))
        
        #放大后
        origin=test_data[1][self.n_seq+self.dis-1:]
        print(output[0], test_data[0][0])
        output=self.renormalize2(output, test_data[0])
        plt.figure()
        #print(len(origin), len(output))
        plt.plot(np.array(output).reshape(-1), label='predict')
        plt.plot(origin, label='goal')
        plt.legend(loc=0)
        #plt.show()

def test():
    x=np.arange(0, 10000, 0.1)
    x=x.reshape(-1, 1)
    tx=np.arange(20000,22000,0.1)
    tx=tx.reshape(-1, 1)
    ty=np.sin(tx)
    y=np.sin(x)
    rnn=RnnModel(is_train=True)
    #rnn.train([np.cos(x), y])
    #rnn.test([np.cos(tx), ty])
