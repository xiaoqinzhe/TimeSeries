import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataproc.datadb import *
from datetime import datetime
from model.residual_rnn2 import ResidualRnn
from model.pretrain_rrnn import PretrainRRnn

default_params={1:{'batch_size':50, 'n_seq':20, 'lstm_size':20, 'num_layers':3, 'keep_prob':1, 
                    'learning_rate':0.01, 'input_size':1, 'output_size':1, 'num_epochs':100},
                2:{'batch_size':50, 'n_seq':20, 'lstm_size':20, 'num_layers':1, 'keep_prob':1, 
                    'learning_rate':0.01, 'input_size':5, 'output_size':1, 'num_epochs':100}}

class BasicRnn:
    def __init__(self, is_train, params):
        self.is_train=is_train
        self.batch_size=params['batch_size']
        self.n_seq=params['n_seq']
        self.lstm_size=params['lstm_size']
        self.num_layers=params['num_layers']
        self.keep_prob=params['keep_prob']
        self.learning_rate=params['learning_rate']
        self.input_size=params['input_size']
        self.output_size=params['output_size']
        self.grad_clip=2

        self.inputs=tf.placeholder(tf.float32, shape=[None, self.n_seq, self.input_size])
        self.targets=tf.placeholder(tf.float32, shape=[None, self.output_size])
        #lstm=tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        #drop=tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        self.cell=tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell() for _ in range(self.num_layers)])
        self.initial_state=self.cell.zero_state(self.batch_size, tf.float32)
        self.cell_inputs=self.inputs
        self.cell_outputs, self.state=tf.nn.dynamic_rnn(self.cell, self.cell_inputs,initial_state=self.initial_state, dtype=tf.float32)
        print(self.cell.variables)
        #self.re_outs=tf.reshape(self.cell_outputs, [-1, self.lstm_size*self.n_seq])
        self.re_outs=self.cell_outputs[:,-1, :]
        print(self.cell_outputs.name)
        w_o=tf.get_variable('weight', [self.lstm_size, self.output_size])
        b_o=tf.get_variable('bias', [self.output_size])
        self.outputs=tf.matmul(self.re_outs, w_o)+b_o
        # attention
        # Wc=tf.get_variable('weighted_c', shape=(self.lstm_size, 1))
        # Wh=tf.get_variable('weighted_h', shape=(self.lstm_size, 1))
        # wcc=tf.matmul(self.state[self.num_layers-1].c, Wc)
        # ms=[tf.nn.tanh(wcc[i]+tf.matmul(self.cell_outputs[i,:,:], Wh)) for i in range(self.batch_size)]
        # self.ws=tf.nn.softmax(ms)
        # print(self.ws.shape)
        # self.union_outputs = tf.reduce_sum(tf.multiply(self.ws, self.cell_outputs),axis=1)
        # print(self.union_outputs.shape)
        # w_o = tf.get_variable('weight', [self.lstm_size, self.output_size])
        # b_o = tf.get_variable('bias', [self.output_size])
        # self.outputs = tf.matmul(self.union_outputs, w_o) + b_o
        #loss and optimizer
        #self.loss=tf.sqrt(tf.reduce_mean(tf.square(self.targets-self.outputs)))
        # self.loss = tf.reduce_mean(tf.square(self.targets - self.outputs))
        self.loss = tf.losses.huber_loss(self.targets, self.outputs, delta=0.3)
        if is_train:           
            #clipping gradients optimizer
            self.tvars=tf.trainable_variables()
            self.grads, _=tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars), self.grad_clip)
            #self.optimizer=tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.grads, self.tvars))
            self.optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def lstm_cell(self):
        lstm=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
        if not self.is_train or self.keep_prob>=1: return lstm
        drop=tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
        return drop

class Model:
    def __init__(self, is_train = False, params=None, savename='default'):
        self.is_train=is_train
        self.save_path='model/rnnmodels/'+savename
        #self.data_size=20000
        if params==None: params=default_params[1]
        self.params=params
        if 'metrics' not in self.params.keys(): self.params['metrics']=[]
        self.num_epochs=params['num_epochs']
        self.initRnn(params)
        self.prestate=None

    def __del__(self):
        self.sess.close()
        tf.reset_default_graph()

    def initRnn(self, params):
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess=tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options))
        #self.sess=tf.Session()
        with tf.variable_scope('rnnmodel', reuse=None):
            self.train_model=eval(params['model_name']+'(is_train=True, params=params)')
        with tf.variable_scope('rnnmodel', reuse=True):
            params['batch_size']=1
            self.eval_model=eval(params['model_name']+'(is_train=False, params=params)')
        #optimizer=tf.train.AdamOptimizer(0.1).minimize(loss)

        #predict
        #self.meandiff=tf.reduce_mean(self.targets-self.outputs)

        self.saver=tf.train.Saver()
        if not self.is_train: self.saver.restore(self.sess, self.save_path)
    
    def get_batches(self, dx, dy, batch_size,shuffle=False):
        data=[]
        for i in range(len(dx)):
            data.append([dx[i], dy[i]])
        if shuffle: np.random.shuffle(data)
        nb=len(data)//batch_size
        for i in range(nb):
            x, y = [], []   
            for j in range(i*batch_size, (i+1)*batch_size):
                x.append(data[j][0])
                y.append(data[j][1])
            yield np.array(x), np.array(y)
    
    def run_epoch(self, session, model, data, train_op, output_log=False, shuffle=False):
        state=session.run(model.initial_state)
        batches=self.get_batches(data[0], data[1], model.batch_size, shuffle)
        total_loss=i=0
        outputs=[]
        for step, (x, y) in enumerate(batches):
            i+=1
            #feed={model.inputs:x, model.targets:y, model.initial_state:state}
            feed={model.inputs:x, model.targets:y}
            outs, lo, state, _ = self.sess.run([model.outputs, model.loss, model.state, train_op], feed_dict=feed)
            total_loss+=lo
            outputs.append(outs[0])
            if output_log and i%1==0:
                print(i,lo)
        return outputs, total_loss/i

    def train(self, x_train, y_train, validation=None, patient=None):
        if not self.is_train: return
        #with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
        self.sess.run(tf.global_variables_initializer())
        i, test_loss=0, 0
        best_loss, count = 100000, 0
        for e in range(self.num_epochs):
            metrics=[]
            _, train_loss=self.run_epoch(self.sess, self.train_model,[x_train, y_train], self.train_model.optimizer, shuffle=True)
            if validation:
                pre_output, test_loss=self.run_epoch(self.sess, self.eval_model, validation, tf.no_op(), shuffle=False)
                for metricf in self.params['metrics']:
                    metrics.append(metricf(pre_output, validation[1]))
            if((e+1)%1==0): print(e, 'epoch: ','train_loss:', train_loss,'test_loss:', test_loss,'metrics:', metrics)
            if validation and patient:
                if test_loss<best_loss:  best_loss, count = test_loss, 0
                else: count+=1
                if count > patient:
                    print('not improve, early stop.')
                    break
        self.saver.save(self.sess, self.save_path)
        return train_loss

    def predictV(self, seq):
        total_loss=i=0
        outputs=[]
        if self.prestate==None: self.prestate=self.sess.run(self.eval_model.initial_state)
        model=self.eval_model
        for step, x in enumerate(seq):
            i+=1
            #feed={model.inputs:[x], model.initial_state:self.prestate}
            feed={model.inputs:[x]}
            output, self.prestate= self.sess.run([model.outputs, model.state], feed_dict=feed)
            outputs.append(output[0])
            # if output_log and i%20==0: 
            #     print(i,lo)
        #print('total_loss ', total_loss/i)
        return outputs

    def test(self, x_test, y_test):
        outputs, loss=self.run_epoch(self.sess, self.eval_model, [x_test, y_test], tf.no_op())
        return outputs, loss

class ProRnn(BasicRnn):
    def __init__(self, is_train, params):
        super(ProRnn, self).__init__(is_train, params)
        self.outputs=tf.nn.softmax(self.outputs)
        self.one_hot=tf.one_hot(tf.argmax(self.outputs,1), self.output_size,1,0)
        self.loss=-tf.reduce_sum(self.targets*tf.log(self.outputs))
        correct_prediction = tf.equal(tf.argmax(self.outputs,1), tf.argmax(self.targets,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

class ProRnnModel(RnnModel):
    def __init__(self, is_train, params, savename):
        super(ProRnnModel, self).__init__(is_train=is_train, params=params, savename=savename)

    def __del__(self):
        super(ProRnnModel, self).__del__()

    def run_epoch(self, session, model, data, train_op, output_log=True):
        state=session.run(model.initial_state)
        batches=self.get_batches(data[0], data[1], model.batch_size)
        total_loss=total_acc=i=0
        outputs=[]
        for step, (x, y) in enumerate(batches):
            i+=1
            feed={model.inputs:x, model.targets:y, model.initial_state:state}
            lo, acc, state, _ = self.sess.run([model.loss, model.accuracy, model.state, train_op], feed_dict=feed)
            total_loss+=lo
            total_acc+=acc
            outputs.append([model.outputs])
            if output_log and i%20==0: 
                print(i,lo,acc)
        return outputs, total_loss/i, total_acc/i

    def train(self, x_train, y_train, validation=None):
        self.sess.run(tf.global_variables_initializer())
        i=test_loss=0
        for e in range(self.num_epochs):
            _, train_loss, train_acc=self.run_epoch(self.sess, self.train_model,[x_train, y_train], self.train_model.optimizer)
            if validation: _, test_loss, test_acc=self.run_epoch(self.sess, self.eval_model, validation, tf.no_op())
            print(e, 'epoch:', train_loss, test_loss, train_acc, test_acc)
        self.saver.save(self.sess, self.save_path)

    def predictV(self, seq):
        total_loss=i=0
        outputs=[]
        if self.prestate==None: self.prestate=self.sess.run(self.eval_model.initial_state)
        model=self.eval_model
        for step, x in enumerate(seq):
            i+=1
            #feed={model.inputs:[x], model.initial_state:self.prestate}
            feed={model.inputs:[x]}
            output, self.prestate= self.sess.run([model.outputs, model.state], feed_dict=feed)
            outputs.append([output])
        return outputs

    def test(self, x_test, y_test):
        outputs, loss, acc=self.run_epoch(self.sess, self.eval_model, [x_test, y_test], tf.no_op())
        print('test:', loss, acc)
        return outputs

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
