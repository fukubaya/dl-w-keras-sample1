# -*- coding: utf-8 -*-

import argparse
import collections
import sys

from keras.models import Sequential
from keras.layers import Dense, Lambda, Embedding
import keras.backend as K
from keras.utils import np_utils
import numpy as np

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "infile",
        type=argparse.FileType("r"))

    return parser

def main(args):
    stations = []
    
    with args.infile as f:
        for s in f.readlines():
            stations.append(s.strip())

    counter = collections.Counter()

    for s in stations:
        chars = [c for c in s]
        for c in chars:
            counter[c] += 1

    char2index = collections.defaultdict(int)
    for cid, char in enumerate(counter):
        char2index[char] = cid

    vocab_size = len(char2index)
    embed_size = 128
    window_size = 1
    index2char = {v: k for k, v in char2index.items()}

    xs, ys = [], []
    
    for s in stations:
        cids = [char2index[char] for char in s]
        for i in range(window_size, len(cids) - window_size):
            x = cids[i - window_size:i] +  cids[i + 1:i+window_size+1]
            y = cids[i]
            xs.append(x)
            ys.append(y)

    # model
    X = np.array(xs)
    Y = np_utils.to_categorical(ys, vocab_size)

    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embed_size,
            embeddings_initializer='glorot_uniform',
            input_length=window_size * 2))
    model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
    model.add(Dense(vocab_size, kernel_initializer='glorot_uniform', 
                activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X, Y, batch_size=16, epochs=10, verbose=1)
    

    weights = model.layers[0].get_weights()[0]
    print(weights.shape)
    print(weights)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
