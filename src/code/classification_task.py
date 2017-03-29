import urllib.request
import pandas as pd
import numpy as np
import collections
import tensorflow as tf
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
response = urllib.request.urlopen(url)
from sklearn.preprocessing import OneHotEncoder
# data = response.read()      # a `bytes` object
# text = data.decode('utf-8') # a `str`; this step can't be used if data is binary

data = pd.read_csv(response, header=None)


def preprocess_for_embedding(data):
    integer_data = data.copy()
    unique_dict = dict()
    index_counter = 0
    for i in range(data.shape[1]):
        tmp = data.iloc[:, i]
        unique_cat = set(tmp)
        num_cat = len(unique_cat)

        for val in unique_cat:
            tmp.replace(val, index_counter, inplace=True)
            index_counter += 1
        integer_data.loc[:,i] = tmp

        unique_dict[i] = (unique_cat, index_counter)
    return integer_data, unique_dict

def preprocess_for_onehot(data):
    integer_data = data.copy()
    unique_dict = dict()
    for i in range(data.shape[1]):
        tmp = data.iloc[:, i]
        unique_cat = set(tmp)

        for idx, val in enumerate(unique_cat):
            tmp.replace(val, idx, inplace=True)

        integer_data.loc[:,i] = tmp

        unique_dict[i] = unique_cat
    onehot = OneHotEncoder().fit_transform(integer_data).toarray()
    return onehot, unique_dict


input_data, unique_dict = preprocess_for_onehot(data)


class Autoencoder():
    def __init__(self, inputs, layer_dim):

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
        w_list.reverse()
        layer_dim.reverse()
        for idx in range(len(layer_dim)-1):
            with tf.variable_scope('decoder_{}'.format(idx)):
                b = tf.get_variable('b', shape=[layer_dim[idx + 1]],
                                    initializer=tf.constant_initializer(0.0))
                layer_outputs.append(tf.add(tf.matmul(layer_outputs[-1],
                                                      tf.transpose(w_list[idx])),
                                            b))

        self.layer_outputs = layer_outputs
        self.output = layer_outputs[-1]
        self.init = tf.global_variables_initializer()

        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(layer_outputs[0] -
                                                           layer_outputs[-1]), 1))
        self.train_op = tf.train.AdamOptimizer()

writer = tf.summary.FileWriter('logdir')
with tf.Session() as sess:
    input = tf.placeholder(tf.float32, shape=[None, input_data.shape[1]])
    with tf.variable_scope('Model', initializer=tf.truncated_normal_initializer()):
        auto = Autoencoder(input, [500, 10])
    sess.run(auto.init)
    sess.run()
    writer.add_graph(graph=tf.get_default_graph())

writer.close()

