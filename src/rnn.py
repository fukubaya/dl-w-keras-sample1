# -*- coding: utf-8 -*-
"""RNNで固有名詞を生成する"""

import argparse
import collections
import random

from keras.models import Sequential
from keras.layers import (
    Dense,
    SimpleRNN,
    LSTM,
    GRU,
    Activation)
from keras.layers.wrappers import Bidirectional
import numpy as np


def create_parser():
    "parser"

    parser = argparse.ArgumentParser()

    # 対象名詞リスト
    parser.add_argument(
        "infile",
        type=argparse.FileType("r"))

    # 隠れ状態次元数
    parser.add_argument(
        "-H",
        "--hidden",
        type=int,
        default=128,
        help="隠れ状態次元数"
    )

    # 入力系列数
    parser.add_argument(
        "-s",
        "--seq",
        type=int,
        default=2,
        help="入力系列数"
    )

    # 学習回数
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=10,
        help="学習回数"
    )

    return parser


def create_model(hidden_size, seq_len, vocab_size):
    "モデルの生成"
    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(hidden_size,
                 return_sequences=False,
                 recurrent_dropout=0.1,
                 unroll=True),
            input_shape=(seq_len, vocab_size)))
    model.add(Dense(vocab_size))
    model.add(Activation("softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam")
    return model


class DataSet(object):
    "データ"

    def __init__(self, f_obj, seq_len=2):
        # read wordsb
        self.words = []
        for w in f_obj.readlines():
            self.words.append(w.strip())
        # shuffle
        random.shuffle(self.words)

        # chars
        chars = set([])
        for w in self.words:
            for c in [c for c in w]:
                chars.add(c)

        # char2index, index2char
        self.char2index = collections.defaultdict(int)
        self.char2index["$"] = 0
        self.char2index["^"] = 1
        for cid, char in enumerate(chars):
            self.char2index[char] = cid + 2
        self.index2char = {v: k for k, v in self.char2index.items()}

        # create training data
        self.seq_len = seq_len
        input_seqs = []
        label_chars = []
        for w in self.words:
            w_mod = "^" + w + "$"
            for i in range(max(0, len(w_mod) - self.seq_len)):
                input_seqs.append(w_mod[i:i + self.seq_len])
                label_chars.append(w_mod[i + self.seq_len])

        X = np.zeros((len(input_seqs), self.seq_len, self.vocab_size), dtype=np.bool)
        Y = np.zeros((len(input_seqs), self.vocab_size), dtype=np.bool)
        for i, input_seq in enumerate(input_seqs):
            for j, ch in enumerate(input_seq):
                X[i, j, self.char2index[ch]] = 1
            Y[i, self.char2index[label_chars[i]]] = 1
        self.X = X
        self.Y = Y

    @property
    def vocab_size(self):
        "語彙数"
        return len(self.char2index)

    def sample_random_char(self):
        "語彙からランダムに1文字取得"
        return self.index2char[np.random.randint(2, self.vocab_size)]

    def predict_random_seq(self, model, max_len=20):
        "学習済モデルから文字列を生成"
        result_chars = []
        rnd_first_char = self.sample_random_char()
        result_chars = [rnd_first_char]
        test_seq = "^" * (self.seq_len - 1) + rnd_first_char

        for i in range(max_len):
            Xtest = np.zeros((1, self.seq_len, self.vocab_size))
            for i, ch in enumerate(test_seq):
                Xtest[0, i, self.char2index[ch]] = 1
            pred = model.predict(Xtest, verbose=0)[0]
            ypred = self.index2char[np.argmax(pred)]
            if ypred == "$":
                break
            result_chars.append(ypred)
            test_seq = test_seq[1:] + ypred
        result = "".join(result_chars)
        if result in self.words:
            # 完全一致
            result = "=" + result
        else:
            # 部分一致
            for w in self.words:
                if result in w or w in result:
                    result = "*" + result
        return result


def main(args):
    # データ読み込み
    data = DataSet(args.infile, args.seq)

    # 隠れ状態次元数
    hidden_size = args.hidden

    # バッチサイズ
    batch_size = 32

    # モデルを生成
    model = create_model(hidden_size,
                         data.seq_len,
                         data.vocab_size)

    # 学習と生成
    num_iteration = args.iter
    num_samples = 20
    for iteration in range(num_iteration):
        print("Iteration #: %d" % (iteration))
        model.fit(data.X, data.Y, batch_size=batch_size, epochs=1)

        for i in range(num_samples):
            print(data.predict_random_seq(model), end="")
            if i % 10 == 9:
                print()
            else:
                print("\t", end="")


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
