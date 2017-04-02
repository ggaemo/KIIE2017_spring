from sklearn.preprocessing import OneHotEncoder
import urllib.request
import pandas as pd
import tensorflow as tf
import numpy as np

def get_data_mushroom(preprocess):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    response = urllib.request.urlopen(url)
    data = pd.read_csv(response, header=None)
    target = data.iloc[:,0]
    target_values = list(set(target))
    target = [target_values.index(x) for x in target]
    target = OneHotEncoder().fit_transform(np.expand_dims(target, 1)).toarray()
    data = data.iloc[:,1:]
    if preprocess == 'embed':
        input_data, unique_dict = preprocess_for_embedding(data)
    elif preprocess == 'onehot':
        input_data, unique_dict = preprocess_for_onehot(data)
    else:
        raise AttributeError

    return input_data, target, unique_dict

def preprocess_for_embedding(data):
    unique_dict = dict()
    index_counter = 0
    for i in range(data.shape[1]):
        tmp = data.iloc[:, i]
        unique_cat = set(tmp)
        for val in unique_cat:
            tmp.replace(val, index_counter, inplace=True)
            index_counter += 1

        unique_dict[i] = (unique_cat, index_counter)
    return data, unique_dict

def preprocess_for_onehot(data):
    unique_dict = dict()
    for i in range(data.shape[1]):
        tmp = data.iloc[:, i]
        unique_cat = set(tmp)

        for idx, val in enumerate(unique_cat):
            tmp.replace(val, idx, inplace=True)

        # integer_data.loc[:,i] = tmp

        unique_dict[i] = unique_cat
    onehot = OneHotEncoder().fit_transform(data).toarray()
    return onehot, unique_dict

def input_producer(raw_data, batch_size, dtype=tf.float32):
    with tf.variable_scope("InputProducer", [raw_data, batch_size]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=dtype)
        producer = tf.train.input_producer(raw_data).dequeue()
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        input_batch = tf.train.shuffle_batch([producer], batch_size, capacity, min_after_dequeue, num_threads=3,
        allow_smaller_final_batch=True)
        return input_batch

def input_target_producer(raw_data_input, raw_data_target, batch_size, dtype=tf.float32):
    with tf.variable_scope("InputTargetProducer", [raw_data_input, raw_data_target, batch_size]):
        raw_data_input = tf.convert_to_tensor(raw_data_input, name="raw_data_input", dtype=dtype)
        raw_data_target = tf.convert_to_tensor(raw_data_target, name="raw_data_target", dtype=dtype)
        input_producer, target_producer = tf.train.slice_input_producer([raw_data_input, raw_data_target]).dequeue()
        input_batch, target_batch = tf.train.shuffle_batch([input_producer, target_producer], batch_size, num_threads=3, allow_smaller_final_batch=True)
        return input_batch, target_batch
