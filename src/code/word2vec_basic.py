import collections
import math
import os
import random
import zipfile
import pickle

import urllib.request

import numpy as np
import tensorflow as tf





def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename


def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0 #dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    if not  os.path.exists('data.pkl'):
        pickle.dump([data, count, dictionary, reverse_dictionary], open('data.pkl', 'wb'))
    return data, count, dictionary, reverse_dictionary


class Generator(object):
    def __init__(self):
        self.data_index = 0

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
            buffer.append(data[self.data_index])
            data_index = (self.data_index + 1) % len(data) # to cycle
        for i in range(batch_size // num_skips):
            target = skip_window # Just the initialized value
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid: #so that targets are not redundant
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i*num_skips + j] = buffer[skip_window] #input is at the center
                labels[i*num_skips + j] = buffer[target] # target is elsewhere
            buffer.append(data[self.data_index]) # buffer pops the first element and adds a
            #  new
            # element
            data_index = (self.data_index + 1) % len(data)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (self.data_index + len(data) - span) % len(data)
        return batch, labels



# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

filename = maybe_download('text8.zip', 31344016)

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000
if not os.path.exists('data.pkl'):
    words = read_data(filename)
    data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
    del words
else:
    data, count, dictionary, reverse_dictionary = pickle.load(open('data.pkl', 'rb'))

gen = Generator()
batch, labels = gen.generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
    reverse_dictionary[labels[i, 0]])


print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

