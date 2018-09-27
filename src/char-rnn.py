# -*- coding: utf-8 -*-

import argparse
import collections
import sys
import random

from gensim.models import word2vec

from keras.models import Sequential
from keras.layers import Dense, Lambda, Embedding, LSTM, GRU, Dropout, TimeDistributed, Activation, Bidirectional
import keras.backend as K
from keras.utils import np_utils
import numpy as np

def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "infile",
        type=argparse.FileType("r"))

    return parser

def print_most_similarity(model, char):
    print("model.most_similar('{}')".format(char))
    print(model.predict_output_word([char]))

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
    char2index["EOS"] = 0
    char2index["GO"] = 1
    for cid, char in enumerate(counter):
        char2index[char] = cid + 2

    index2char = {v: k for k, v in char2index.items()}
    vocab_size = len(char2index)

    embed_size = 32
    hidden_layer = 512
    batch_size = 1
    seq_len=5

    X_list = []
    Y_list = []
    for s in stations:
        cids = [char2index[char] for char in s] + [char2index["EOS"]]
        #cids = [char2index[char] for char in s]
        for i in range(len(cids) - 1):
            x = cids[i]
            y = cids[i + 1]
            X_list.append([x])
            Y_list.append([np_utils.to_categorical(y, vocab_size)])

    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=1, batch_size=batch_size))
    layer_size = 1
    for i in range(layer_size):
        model.add(Bidirectional(GRU(hidden_layer, return_sequences=True, stateful=True)))
        model.add(Dropout(0.2))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    for e in range(100):
        print("e = {}".format(e))
        #random.shuffle(stations)
        #for s in stations[:int(len(stations)/2)]:
        #for s in stations:
        for i_s, s in enumerate(stations):
            cids = [char2index["GO"]] + [char2index[char] for char in s] + [char2index["EOS"]]
            for i in range(len(cids) - 1):
                x = cids[i]
                y = cids[i + 1]
                X = np.array([x])
                Y = np.array([[np_utils.to_categorical(y, vocab_size)]])
                loss = model.train_on_batch(X, Y)
                #print('loss = {}'.format(loss))
            model.reset_states()
#        for epoch in range(int(len(X_list) / batch_size) - 1):
#            #    for epoch in range(1000):
#            i_start = epoch * batch_size
#            i_end = (epoch + 1) * batch_size
#            X = np.array(X_list[i_start:i_end])
#            Y = np.array(Y_list[i_start:i_end])
#            loss = model.train_on_batch(X, Y)
#            print('loss = {}'.format(loss))
#            model.reset_states()

            model.reset_states()
            if i_s % 100 == 0:
                print("e = {}, s = {}".format(e, i_s))
                top_n = np.flip(model.predict_on_batch(np.array([char2index["GO"]] * batch_size))[0][0].argsort())[:10]
                results = []
                for idx_init in top_n:
#                for i in range(10):
                    #idx_init = np.random.randint(vocab_size)
                    #idx_init = char2index["GO"]
                    idx_current = idx_init
                    idx_result = [idx_init]
                    for j in range(10):
                        idx_current = model.predict_on_batch(np.array([idx_current] * batch_size))[0].argmax()
                        if idx_current == char2index["EOS"]:
                            break
                        idx_result.append(idx_current)
            
                    results.append("".join([index2char[idx] for idx in idx_result]))
                print("\t".join(results))


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
