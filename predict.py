# coding:utf-8
import random
import os
from keras.models import load_model
import numpy as np
from config import Config
from prepare_data import *


class Predict(object):
    def __init__(self, config):
        self.config = config
        self.word2numF, self.num2word, self.words, self.poems = prepare_data()

        self.poem = self.poems.split(']')
        self.poem_num = len(self.poem)
        if os.path.exists(self.config.model_path):
            self.model = load_model(self.config.model_path)

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

    def predict_first(self, char, diversity=1.0):
        '''给定第一个字'''
        index = random.randint(0, self.poem_num)
        sentence = char + self.poem[index][1: self.config.max_len]
        generate = str(char)
        generate += self._preds(sentence, length=23, diversity=diversity)
        return generate

    def predict_hide(self, chars, diversity=1.0):
        '''给定四个字'''
        generate = ''
        for i in range(4):
            index = random.randint(0, self.poem_num)
            sentence = chars[i] + self.poem[index][1: self.config.max_len]
            gene = str(chars[i])
            gene += self._preds(sentence, length=5, diversity=diversity)
            gene = gene[:-1]
            if i == 0 or i == 2:
                gene += '，'
            else:
                gene += '。'
            generate += gene
        return generate


def main():
    pred = Predict(Config)
    print('model loaded!\n')
    while True:
        option = int(input('可选功能如下：\n'
                           '\b1.随机生成\n'
                           '\b2.输入首字\n'
                           '\b3.输入第一句（包括逗号）\n'
                           '\b4.藏头诗\n'
                           '\b5.退出\n'
                           '请输入功能序号：'))
        if option == 1:
            predict = pred.predict_random()
            print('\n' + predict + '\n')
        elif option == 2:
            char = input('请输入首字：')
            predict = pred.predict_first(char)
            print('\n' + predict + '\n')
        elif option == 3:
            text = input('请输入第一句（包括逗号）：')
            predict = pred.predict_sen(text)
            print('\n' + predict + '\n')
        elif option == 4:
            text = input('请输入四个字：')
            predict = pred.predict_hide(text)
            print('\n' + predict + '\n')
        else:
            break


if __name__ == '__main__':
    main()
