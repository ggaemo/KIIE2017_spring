import urllib.request
import os
import pandas as pd
import numpy as np
import collections
import tensorflow as tf

from reader import preprocess_for_onehot, preprocess_for_embedding, get_data


class Autoencoder():
    def __init__(self, inputs, layer_dim, category_dim, squared_loss):

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

        if squared_loss:
            self.loss = tf.reduce_mean(
                tf.losses.mean_squared_error(inputs, layer_outputs[-1], scope='mse'))
            tf.summary.scalar('mse', self.loss)
        else:
            for idx, cat_output_dim in enumerate(category_dim):
                with tf.variable_scope('category_{}'.format(idx)):
                    W = tf.get_variable('W', shape=[layer_dim[1], 1])
                    b = tf.get_variable('b', shape=[layer_dim[idx + 1]],
                                        initializer=tf.constant_initializer(0.0))
                    layer_outputs.append(tf.add(tf.matmul(layer_outputs[-1],
                                                          tf.transpose(w_list[idx])), b))

        self.layer_outputs = layer_outputs

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)
        self.init = tf.global_variables_initializer()
        self.merged_summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()

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
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step =self.global_step)
        self.prediction = tf.argmax(layer_outputs[-1], axis=1)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(target,
                                                                           axis=1)),
                                tf.float32))
        self.init = tf.global_variables_initializer()


def train_autoencoder(max_epoch):

    input_data, target, unique_dict = get_data('onehot')


    hidden_dim = 500
    writer = tf.summary.FileWriter('logdir_{}'.format(hidden_dim))


    with tf.Session() as sess:
        input_placeholder = tf.placeholder(tf.float32, shape=[None, input_data.shape[1]])
        with tf.variable_scope('Model', initializer=tf.truncated_normal_initializer()):
            auto = Autoencoder(input_placeholder, [hidden_dim], [len(x) for x in
                                                        unique_dict.values()],
                               True)
        sess.run(auto.init)
        for i in range(max_epoch):
            loss, _, summary = sess.run([auto.loss, auto.train_op, auto.merged_summary,
                                         ], feed_dict={input_placeholder:input_data})

            print('Epoch {} loss: {}'.format(i, loss))
            writer.add_summary(summary, tf.train.global_step(sess, auto.global_step))

        hidden = sess.run(auto.hidden, feed_dict={input_placeholder: input_data})

        np.save('./logdir_{0}/autoencoder_representation.npy'.format(hidden_dim), hidden)
        hidden_var = tf.Variable(hidden)
        sess.run(tf.global_variables_initializer())

        auto.saver.save(sess, 'logdir_{}/weights'.format(hidden_dim),
                        global_step=tf.train.global_step(sess, auto.global_step))
        writer.add_graph(graph=tf.get_default_graph())

    writer.close()

def train_classifier(max_epoch):
    input_data, target, unique_dict = get_data('onehot')

    input_data = np.load('./logdir_{0}/autoencoder_representation.npy'.format(
        500))

    with tf.Session() as sess:
        input_placeholder = tf.placeholder(tf.float32, shape=[None, input_data.shape[1]])
        target_placeholder = tf.placeholder(tf.int32, shape=[None, target.shape[1]])
        net = NeuralNetwork(input_placeholder, target_placeholder, [5, 10, 10, target.shape[1]])
        sess.run(net.init)

        for i in range(max_epoch):
            loss, _ = sess.run([net.loss, net.train_op], feed_dict={input_placeholder:
                                                                        input_data,
                                                                    target_placeholder:
                                                                        target})
            if i % 100 == 0:
                acc = sess.run(net.accuracy, feed_dict={input_placeholder:
                                                      input_data,
                                                  target_placeholder:
                                                      target})
                print('Epoch : {0} loss: {1} accuracy : {2}'.format(i, loss, acc))


def managed_train(max_epoch):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    response = urllib.request.urlopen(url)
    data = pd.read_csv(response, header=None)

    input_data, unique_dict = preprocess_for_onehot(data)
    producer = tf.train.input_producer(input_data)
    input_batch = tf.train.batch(producer, 32)
    with tf.Graph().as_default():
        input = tf.placeholder(tf.float32, shape=[None, input_data.shape[1]])
        initializer = tf.truncated_normal_initializer()
        with tf.variable_scope('Model', initializer=initializer):
            auto = Autoencoder(input_batch, [500], [len(x) for x in unique_dict.values()],
                               True)

        sv = tf.train.Supervisor(logdir='logdir_sv')
        with sv.managed_session() as sess:

            for i in range(max_epoch):
                loss, _, summary = sess.run([auto.loss, auto.train_op],
                                    feed_dict={input:input_data})
                print('Epoch {} loss: {}'.format(i, loss))

            sv.saver.save(sess, 'save', global_step=sv.global_step)

def embed(max_epoch):
    input_data, target, unique_dict = get_data('embed')
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

if __name__ == '__main__':
    train_classifier(1000)
    # train(200)
    # managed_train(20000)