import urllib.request
import os
import pandas as pd
import numpy as np
import collections
import tensorflow as tf
import reader
from reader import preprocess_for_onehot, preprocess_for_embedding, get_data_mushroom, input_producer

class InputMushroom(object):
  """The input data."""
  def __init__(self, batch_size=None):
      input_data, target, unique_dict = reader.get_data_mushroom('onehot')
      if batch_size:
          self.input_data = reader.input_producer(input_data, batch_size)
      else:
          self.input_data = reader.input_producer(input_data, len(input_data))


class InputTargetMushroom(object):
    def __init__(self, batch_size):
        input_data, target, unique_dict = reader.get_data_mushroom('onehot')
        self.input_data, self.target = reader.input_target_producer(input_data, target, batch_size)

class InputTargetMushroomAE():
    def __init__(self, batch_size, data_path):
        _, target, unique_dict = reader.get_data_mushroom('onehot')
        input_data = np.load(data_path)
        if batch_size:
            self.input_data, self.target = reader.input_target_producer(input_data, target, batch_size)
        else:
            self.input_data, self.target = reader.input_target_producer(input_data, target, len(input_data))


class Autoencoder():
    def __init__(self, inputs, layer_dim, is_training):

        input_dim = inputs.get_shape()[1]
        if not isinstance(layer_dim, collections.Iterable):
            layer_dim = list(layer_dim)
        layer_dim.insert(0, input_dim)

        layer_outputs = [inputs]
        w_list = list()
        for idx, layer_input_dim in enumerate(layer_dim[:-1]):
            with tf.variable_scope('encoder_{}'.format(idx)):
                W = tf.get_variable('W', shape=[layer_input_dim, layer_dim[idx+1]])
                b = tf.get_variable('b', shape=[layer_dim[idx+1]],
                                    initializer=tf.constant_initializer(0.0))
                layer_outputs.append(tf.add(tf.matmul(layer_outputs[-1], W), b))
                w_list.append(W)

        self.hidden = layer_outputs[-1]

        w_list.reverse()
        layer_dim.reverse()
        for idx, layer_input_dim in enumerate(layer_dim[:-1]):
            with tf.variable_scope('decoder_{}'.format(idx)):
                b = tf.get_variable('b', shape=[layer_dim[idx + 1]],
                                    initializer=tf.constant_initializer(0.0))
                layer_outputs.append(tf.add(tf.matmul(layer_outputs[-1],
                                                      tf.transpose(w_list[idx])), b))


        self.loss = tf.reduce_mean(
            tf.losses.mean_squared_error(inputs, layer_outputs[-1], scope='mse'))
        tf.summary.scalar('mse', self.loss)

            # for idx, cat_output_dim in enumerate(category_dim):
            #     with tf.variable_scope('category_{}'.format(idx)):
            #         W = tf.get_variable('W', shape=[layer_dim[1], 1])
            #         b = tf.get_variable('b', shape=[layer_dim[idx + 1]],
            #                             initializer=tf.constant_initializer(0.0))
            #         layer_outputs.append(tf.add(tf.matmul(layer_outputs[-1],
            #                                               tf.transpose(w_list[idx])), b))

        self.layer_outputs = layer_outputs

        # self.global_step = tf.Variable(0, trainable=False, name='global_step')
        if is_training:
            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        # self.init = tf.global_variables_initializer()
        # self.merged_summary = tf.summary.merge_all()
        # self.saver = tf.train.Saver()

class Embedder():
    def __init__(self, inputs, embed_dim, input_dim):
        number_cols = inputs.get_shape()[1]

        embeddings = tf.get_variable('embedding', shape=[input_dim, embed_dim],
                                     initializer=tf.constant_initializer(3))
        self.embed = tf.nn.embedding_lookup(embeddings, inputs)
        flatten = tf.reshape(self.embed, [None, number_cols * embed_dim])

        self.init = tf.global_variables_initializer()

class NeuralNetwork():
    def __init__(self, inputs, target, layer_dim):

        input_dim = inputs.get_shape()[1]
        if not isinstance(layer_dim, collections.Iterable):
            layer_dim = list(layer_dim)
        layer_dim.insert(0, input_dim)

        layer_outputs = [inputs]
        for idx, layer_input_dim in enumerate(layer_dim[:-1]):
            with tf.variable_scope('layer_{}'.format(idx)):
                W = tf.get_variable('W', shape=[layer_input_dim, layer_dim[idx+1]])
                b = tf.get_variable('b', shape=[layer_dim[idx+1]],
                                    initializer=tf.constant_initializer(0.0))
                layer_outputs.append(tf.add(tf.matmul(layer_outputs[-1], W), b))

        self.loss = tf.losses.softmax_cross_entropy(target, layer_outputs[-1])
        self.global_step = tf.Variable(0, trainable=False, name ='global_step')
        self.prediction = tf.argmax(layer_outputs[-1], axis=1)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(target,axis=1)),tf.float32))

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)
        self.init = tf.global_variables_initializer()


def train_autoencoder(max_epoch, hidden_dim, batch_size, data):
    print('training autoencoder on {}'.format(repr(data)))
    log_dir = 'logdir_autoencoder_{}'.format(hidden_dim)
    with tf.Graph().as_default():
        # input_batch = InputMushroom(batch_size)
        # input_batch_eval = InputMushroom()
        input_batch = data(batch_size)
        input_batch_eval = data()
        with tf.variable_scope('Model', initializer=tf.truncated_normal_initializer(), reuse=None):
            train_model = Autoencoder(input_batch.input_data, [hidden_dim],True)

        with tf.variable_scope('Model', initializer=tf.truncated_normal_initializer(), reuse=True):
            eval_model = Autoencoder(input_batch_eval.input_data, [hidden_dim],False)

        sv = tf.train.Supervisor(logdir=log_dir)
        with sv.managed_session() as sess:
            for i in range(max_epoch):
                loss, _ = sess.run([train_model.loss, train_model.train_op])

                print('Epoch {} loss: {}'.format(i, loss))

            hidden = sess.run(eval_model.hidden)
            np.save(os.path.join(log_dir, 'autoencoder_representation.npy'.format(hidden_dim)), hidden)

            sv.saver.save(sess, os.path.join(log_dir, 'weights'.format(hidden_dim)),sv.global_step)


def train_classifier(max_epoch, layer_dim, batch_size, data):
    print('training classifier on {}'.format(repr(data)))
    log_dir = 'classifier_{}'.format(str(layer_dim))

    with tf.Graph().as_default():
        # input_batch, target_batch = InputTargetMushroom(batch_size)
        # input_batch_eval, target_batch_eval = InputTargetMushroom()

        input_batch, target_batch = data(batch_size)
        input_batch_eval, target_batch_eval = data()
        layer_dim.append(target_batch.get_shape()[1]) #added for classification purposes
        with tf.variable_scope('Classifier', initializer=tf.truncated_normal_initializer(), resuse=None):
            net = NeuralNetwork(input_batch, target_batch, layer_dim)

        with tf.variable_scope('Classifier', initializer=tf.truncated_normal_initializer(), resuse=True):
            net_eval = NeuralNetwork(input_batch_eval, target_batch_eval, layer_dim)
        sv = tf.train.Supervisor(logdir=log_dir)

        with sv.managed_session() as sess:

            for i in range(max_epoch):
                sess.run([net.loss, net.train_op])
                if i % 100 == 0:
                    loss, acc = sess.run([net.loss, net.accuracy])
                    eval_acc = sess.run(net_eval.accuracy)
                    print('Epoch : {0} train loss: {1} eval accuracy'.format(i, loss, eval_acc))
            sv.saver.save(sess, os.path.join(log_dir, 'weights'), global_step=sv.global_step)

def embed(max_epoch):
    input_data, target, unique_dict = get_data_mushroom('embed')
    hidden_dim = 500
    writer = tf.summary.FileWriter('logdir_{}'.format(hidden_dim))



    category_dim = [x[1] for x in unique_dict.values()]

    print(category_dim)


    with tf.Session() as sess:
        input_placeholder = tf.placeholder(tf.int32, shape=[None, input_data.shape[1]])
        with tf.variable_scope('Model', initializer=tf.truncated_normal_initializer()):
            emb = Embedder(input_placeholder, 200, 119)
        sess.run(emb.init)
        for i in range(max_epoch):
            a = sess.run(emb.embed, feed_dict={input_placeholder:input_data.values[1:2,:]})
            print(a.shape)
            print(a)

            writer.add_summary(summary, tf.train.global_step(sess, auto.global_step))

        hidden = sess.run(auto.hidden, feed_dict={input_placeholder: input_data[:10]})
        hidden_var = tf.Variable(hidden[:, :10])
        sess.run(tf.global_variables_initializer())

        auto.saver.save(sess, 'logdir_{}/weights'.format(hidden_dim),
                        global_step=tf.train.global_step(sess, auto.global_step))
        writer.add_graph(graph=tf.get_default_graph())

    writer.close()




