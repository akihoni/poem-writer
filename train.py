# coding:utf-8
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, ModelCheckpoint
import os
import numpy as np
import random
from config import Config
from prepare_data import *


class Model(object):
    def __init__(self, Config):
        self.config = Config
        self.model = None
        self.word2numF, self.num2word, \
        self.words, self.poems \
            = prepare_data()
        self.poem = self.poems.split(']')
        self.poem_num = len(self.poem)
        if os.path.exists(self.config.model_path):
            self.model = load_model(self.config.model_path)
        else:
            self.train()

    def sample(self, preds, diversity=1.0):
        preds = np.asarray(preds).astype('float64')
        exp_preds = np.power(preds, 1. / diversity)
        preds = exp_preds / np.sum(exp_preds)
        pro = np.random.choice(range(len(preds)), 1, p=preds)
        return int(pro.squeeze())

    def _preds(self, sentence, length=18, diversity=1.0):
        '''生成长度为length的字符串'''
        sentence = sentence[:self.config.max_len]
        generate = ''
        for i in range(length):
            pred = self._pred(sentence, diversity)
            generate += pred
            sentence = sentence[1:] + pred
        return generate

    def _pred(self, sentence, diversity=1.0):
        '''预测下一个字符'''
        sentence = sentence[-self.config.max_len:]
        x_pred = np.zeros(shape=(1, self.config.max_len),
                          dtype=np.int32)
        for t, char in enumerate(sentence):
            x_pred[0, t] = self.word2numF(char)
        preds = self.model.predict(x_pred)[0]
        next_index = self.sample(preds, diversity=diversity)
        next_char = self.num2word[next_index]
        return next_char

    def predict_sen(self, sen, diversity=1.0):
        '''给定第一句（前6个字符）'''
        sentence = sen[-self.config.max_len:]
        generate = str(sentence)
        generate += self._preds(sentence,
                                length=self.config.poem_len - self.config.max_len,
                                diversity=diversity)
        return generate

    def predict_random(self, diversity=1.0):
        '''随机生成，此处随机取第一句'''
        index = random.randint(0, self.poem_num)
        sentence = self.poem[index][:self.config.max_len]
        generate = self.predict_sen(sentence, diversity=diversity)
        return generate

    def generate_sample(self, epoch, logs):
        if epoch % 10 != 0:
            return
        with open(self.config.output_path, 'a', encoding='utf-8') as file:
            file.write('=============Epoch {}============='.format(epoch))
        print('\n=============Epoch {}============='.format(epoch) + '\n')
        for diversity in [0.5, 1.0, 1.5]:
            print('-------------Diversity {}-------------'.format(diversity))
            generate = self.predict_random(diversity=diversity)
            print(generate)
            with open(self.config.output_path, 'a', encoding='utf-8') as file:
                file.write(generate + '\n')

    def data_generator(self):
        i = 0
        while 1:
            x = self.poems[i: i + self.config.max_len]
            y = self.poems[i + self.config.max_len]
            if ']' in x or ']' in y:
                i += 1
                continue

            x_vec = np.zeros(shape=(1, self.config.max_len),
                             dtype=np.int32)
            for t, char in enumerate(x):
                x_vec[0, t] = self.word2numF(char)

            y_vec = np.zeros(shape=(1, len(self.words)),
                             dtype=np.bool)
            y_vec[0, self.word2numF(y)] = 1

            yield x_vec, y_vec
            i += 1

    def train(self):
        epoch_num = len(self.poems) // self.config.batch_size

        if not self.model:
            print('building model...')
            self.model = Sequential()
            self.model.add(Embedding(len(self.words) + 2,
                                     300,
                                     input_length=self.config.max_len))
            self.model.add(LSTM(512,
                                return_sequences=True))
            self.model.add(Dropout(0.6))
            self.model.add(LSTM(256))
            self.model.add(Dropout(0.6))
            self.model.add(Dense(len(self.words),
                                 activation='softmax'))
            optimizer = Adam(lr=self.config.learning_rate)
            self.model.compile(optimizer=optimizer,
                               loss='categorical_crossentropy')
        self.model.summary()

        print('training model...')
        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=self.config.epoch_num,
            callbacks=[
                ModelCheckpoint(self.config.model_path, save_weights_only=False),
                LambdaCallback(on_epoch_end=self.generate_sample)
            ]
        )


def main():
    model = Model(Config)


if __name__ == '__main__':
    main()
