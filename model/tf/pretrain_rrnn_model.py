from model.seq_rnn_model2 import RnnModel
import tensorflow as tf


class PretrainRRnnModel(RnnModel):
    def __init__(self, is_train=False, params=None, savename='default'):
        super(PretrainRRnnModel, self).__init__(is_train, params, savename)
        self.default_scope_scales = [[1,1]]*(self.eval_model.num_scopes - 1)
        writer = tf.summary.FileWriter("./logs", tf.get_default_graph())
        writer.close()

    def __del__(self):
        super(PretrainRRnnModel, self).__del__()

    def run_epoch(self, session, model, data, train_op, scope_scales, output_log=False, shuffle=False):
        # state=session.run(model.initial_state)
        batches = self.get_batches(data[0], data[1], model.batch_size, shuffle=shuffle)
        total_loss = i = 0
        outputs = []
        for step, (x, y) in enumerate(batches):
            i += 1
            # feed={model.inputs:x, model.targets:y, model.initial_state:state}
            feed = {model.targets: y.tolist()}
            for j in range(model.num_scopes - 1): feed[model.scope_scales[j]] = scope_scales[j]
            for j in range(model.num_scopes): feed[model.inputs[j]] = x[:, j].tolist()
            # lo, state, _ = self.sess.run([model.loss, model.state, train_op], feed_dict=feed)
            outs, lo, _ = self.sess.run([model.outputs, model.loss, train_op], feed_dict=feed)
            total_loss += lo
            outputs.append(outs[0])
            if output_log and i % 20 == 0:
                print(i, lo)
        return outputs, total_loss / i

    def saveModel(self, session, step):
        saver=tf.train.Saver()
        saver.save(session, 'saved_model/pretrain_model',global_step=step)

    def loadModel(self, session, step):
        saver=tf.train.Saver()
        saver.restore(session, 'saved_model/pretrain_model-{}'.format(step))

    def pretrain(self, x_train, y_train, epoch=None, patients=None, validation=None):
        # with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
        if epoch==None: epoch=self.num_epochs
        if patients==None: patients=[20]*self.train_model.num_scopes
        self.sess.run(tf.global_variables_initializer())
        test_loss = train_loss = 0
        for layer in range(self.train_model.num_scopes):
            best_loss, notimproved_count=100000, 0
            if layer==0: var_scope_scales = [[1, 0]]*(self.train_model.num_scopes-1)

            print('the %dst lstm layer training...' % layer)
            for e in range(epoch):
                if layer and e < 100:
                    #var_scope_scales[layer-1][1]+=(1+e)/100
                    var_scope_scales[layer - 1][1] = 1
                metrics = []
                # if e<=p1: var_scope_scales=[1, 1]
                # elif e<=p2: var_scope_scales=[0.2, 1]
                # else:  var_scope_scales=[0.2, 0.2]
                _, train_loss = self.run_epoch(self.sess, self.train_model, [x_train, y_train], self.train_model.bp_operation,
                                               var_scope_scales, shuffle=True)
                tf.summary.scalar('train_loss', train_loss)
                if validation:
                    pre_output, test_loss = self.run_epoch(self.sess, self.eval_model, validation, tf.no_op(),
                                                           var_scope_scales, shuffle=False)
                    tf.summary.scalar('test_loss', test_loss)
                    for metricf in self.params['metrics']:
                        metrics.append(metricf(pre_output, validation[1]))
                    if ((e + 1) % 1 == 0): print(e, 'epoch: ', 'train_loss:', train_loss, 'test_loss:', test_loss,
                                                     'metrics:', metrics)
                    if e>100:
                        if test_loss<best_loss:
                            best_loss=test_loss
                            notimproved_count=0
                        else: notimproved_count+=1
                    if notimproved_count>patients[layer]:
                        print('not improve, break')
                        break
        self.saver.save(self.sess, self.save_path)
        return train_loss

    def pretrain2(self, x_train, y_train, epoch=None, patients=None, validation=None):
        # with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options)) as sess:
        if epoch==None: epoch=self.num_epochs
        if patients==None: patients=[20]*self.train_model.num_scopes
        self.sess.run(tf.global_variables_initializer())
        test_loss = train_loss = 0
        for layer in range(self.train_model.num_scopes):
            best_loss, notimproved_count=100000, 0
            if layer==0: var_scope_scales = [[1, 0]]*(self.train_model.num_scopes-1)

            print('the %dst lstm layer training...' % layer)
            for e in range(epoch):
                if layer and e < 100:
                    #var_scope_scales[layer-1][1]+=(1+e)/100
                    var_scope_scales[layer - 1][1] = 1
                metrics = []
                # if e<=p1: var_scope_scales=[1, 1]
                # elif e<=p2: var_scope_scales=[0.2, 1]
                # else:  var_scope_scales=[0.2, 0.2]
                _, train_loss = self.run_epoch(self.sess, self.train_model, [x_train, y_train], self.train_model.bp_operation,
                                               var_scope_scales, shuffle=True)
                tf.summary.scalar('train_loss', train_loss)
                if validation:
                    pre_output, test_loss = self.run_epoch(self.sess, self.eval_model, validation, tf.no_op(),
                                                           var_scope_scales, shuffle=False)
                    tf.summary.scalar('test_loss', test_loss)
                    for metricf in self.params['metrics']:
                        metrics.append(metricf(pre_output, validation[1]))
                    if ((e + 1) % 1 == 0): print(e, 'epoch: ', 'train_loss:', train_loss, 'test_loss:', test_loss,
                                                     'metrics:', metrics)
                    if e>100:
                        if test_loss<best_loss:
                            best_loss=test_loss
                            notimproved_count=0
                        else: notimproved_count+=1
                    if notimproved_count>patients[layer]:
                        print('not improve, break')
                        break
        # self.saveModel(self.sess, pretrain_layer)
        return train_loss

    def test(self, x_test, y_test):
        outputs, loss = self.run_epoch(self.sess, self.eval_model, [x_test, y_test], tf.no_op(),
                                       self.default_scope_scales)
        return outputs, loss

    def predictV(self, seq):
        total_loss = i = 0
        outputs = []
        # if self.prestate==None: self.prestate=self.sess.run(self.eval_model.initial_state)
        model = self.eval_model
        for step, x in enumerate(seq):
            i += 1
            feed = {}
            for j in range(model.num_scopes - 1): feed[model.scope_scales[j]] = self.default_scope_scales[j]
            for j in range(model.num_scopes): feed[model.inputs[j]] = [x[j]]
            output = self.sess.run([model.outputs[-1]], feed_dict=feed)
            outputs.append(output[0])
        # if output_log and i%20==0:
        #	 print(i,lo)
        # print('total_loss ', total_loss/i)
        return outputs