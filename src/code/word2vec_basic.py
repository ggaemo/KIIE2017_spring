import collections
import math
import os
import random
import zipfile
import pickle

import urllib.request

import numpy as np
import tensorflow as tf


class DataGenerator(object):
    def __init__(self):
        self.data_index = 0

    def read_data(self, url='http://mattmahoney.net/dc/', filename='text8.zip'):

        if not os.path.exists('data.pkl'):

            def maybe_download(url, filename, expected_bytes):
                """Download a file if not present, and make sure it's the right size."""
                if not os.path.exists(filename):
                    print('downloading file')
                    filename, _ = urllib.request.urlretrieve(url + filename, filename)
                statinfo = os.stat(filename)
                if statinfo.st_size == expected_bytes:
                    print('\tFound and verified', filename)
                else:
                    print(statinfo.st_size)
                    raise Exception(
                        'Failed to verify ' + filename + '. Can you get to it with a '
                                                          'browser?')
                return filename

            filename = maybe_download(url, filename, 31344016)

            print('Reading file')
            """Extract the first file enclosed in a zip file as a list of words"""
            with zipfile.ZipFile(filename) as f:
                data = tf.compat.as_str(f.read(f.namelist()[0])).split()
                self.words = data
            self.build_dataset()
        else:
            data, count, dictionary, reverse_dictionary = \
                pickle.load(open('data.pkl', 'rb'))

            self.data = data
            self.count = count
            self.dictionary = dictionary
            self.reverse_dictionary = reverse_dictionary

        print('loaded data, count,  dictionary, reverse dictionary')

    def build_dataset(self, vocabulary_size = 50000):
        count = [['UNK', -1]]
        count.extend(collections.Counter(self.words).most_common(vocabulary_size - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0
        for word in self.words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        if not os.path.exists('data.pkl'):
            pickle.dump([data, count, dictionary, reverse_dictionary],
                        open('data.pkl', 'wb'))
        self.data = data
        self.count = count
        self.dictionary,= dictionary
        self.reverse_dictionary = reverse_dictionary

    def generate_batch(self, batch_size, num_skips, skip_window):
        '''
        generate batch data for the Skip-gram Model
        deque(DECK) works as a queue and a stack, where you can insert and pop whereever
        you like, if the manlen excedes, it will pop
        :param batch_size:
        :param num_skips: How many times to reuse an input to generate a label
        :param skip_window: how many words to consider left and right
        :return:
        '''

        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 #[skip_window target skip_window]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data) # to cycle
        for i in range(batch_size // num_skips):
            target = skip_window # Just the initialized value
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid: #so that targets are not redundant
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i*num_skips + j] = buffer[skip_window] #input is at the center
                labels[i*num_skips + j] = buffer[target] # target is elsewhere
            buffer.append(self.data[self.data_index]) # buffer pops the first element and
            # adds a
            #  new
            # element
            self.data_index = (self.data_index + 1) % len(self.data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels


    def sample(self):
        print('showing samples')
        batch, labels = self.generate_batch(8, 2, 1)
        for i in range(8):
            print(batch[i], self.reverse_dictionary[batch[i]], '->', labels[i, 0],
                  self.reverse_dictionary[labels[i, 0]])


class Word2VecBasic():
    def __init__(self, batch_size, embedding_size, skip_window, num_skips,
                 vocabulary_size, num_sampled):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.vocabulary_size = vocabulary_size
        self.num_sampled = num_sampled


    def build_model(self, inputs, train):

        with tf.device('/cpu:0'):
            embeddings = tf.get_variable('embedding', shape=[self.vocabulary_size],
                                         initializer=tf.truncated_normal_initializer())
            embed = tf.nn.embedding_lookup(embeddings, inputs.input_data)

            nce_weights = tf.get_variable('nce_weights', shape=[self.vocabulary_size,
                                                                self.embedding_size],
                                          initializer=tf.truncated_normal_initializer())
            nce_bias = tf.get_variable('nce_bias', shape=[self.vocabulary_size],
                                       initializer=tf.constant_initializer())

        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights,
                           nce_bias,
                           inputs.targets,
                           embed,
                           self.num_sampled,
                           self.vocabulary_size), name='nce_loss'
        )

        self.optimizer = tf.train.AdamOptimizer().minimize(loss)

        if not train:
            #cosine similarity check
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, inputs.input_data)
            self.similarity = tf.matmul(valid_embeddings, normalized_embeddings,
                                    transpose_b=True)


def train(max_epoch):

    gen = DataGenerator()
    gen.read_data()

    class example():
        def __int__(self):
            self.input_data = tf.placeholder(tf.int32, shape=[None])
            self.targets = tf.placeholder(tf.int32, shape=[None, 1])


    with tf.Graph().as_default():
        train_data = example()
        with tf.variable_scope('Train'):
            model = Word2VecBasic(8, 64, 3, 2, 50000, 15)
            model.build_model(train_data, True)
        with tf.variable_scope('Valid', reuse=True):
            valid_model = Word2VecBasic(8, 64, 3, 2, 50000, 15)
            valid_model.build_model(train_data, False)

        sv = tf.train.Supervisor(logdir='save')
        with sv.managed_session() as sess:
            for i in range(max_epoch):

                input, label = gen.generate_batch(8, 3, 2)
                sess.run(model.optimizer, feed_dict={train_data.input_data: input,
                                                     train_data.targets: label})

                if i % 10 == 0:
                    sess.run(valid_model.similarity, )









def test():
    gen = DataGenerator()
    gen.read_data()
    gen.sample()
    batch, labels = gen.generate_batch(batch_size=8, num_skips=2, skip_window=1)


    print('Most common words (+UNK)', gen.count[:5])
    print('Sample data', gen.data[:10], [gen.reverse_dictionary[i] for i in gen.data[:10]])

