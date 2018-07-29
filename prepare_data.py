# coding:utf-8
from config import *


def prepare_data():
    # 读取并分割数据集内容
    poems = ''
    with open(Config.data_set, 'r', encoding='utf-8') as file:
        for line in file:
            text = line.strip() + ']'
            text = text.split(':')[1]
            if len(text) <= 5:
                continue
            if text[5] == '，':
                poems += text

    # 对每个字计数
    words = sorted(list(poems))
    words_counter = {}
    for word in words:
        if word in words_counter:
            words_counter[word] += 1
        else:
            words_counter[word] = 1

    # 去掉低频字
    erase = []
    for word in words_counter:
        if words_counter[word] <= 2:
            erase.append(word)
    for word in erase:
        del words_counter[word]

    # 补充空格
    word_pairs = sorted(words_counter.items(), key=lambda x: x[1], reverse=True)
    words, _ = zip(*word_pairs)
    words += (' ',)

    # 建立映射
    word2num = dict((c, i) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, len(words) - 1)
    num2word = dict((i, c) for i, c in enumerate(words))

    return word2numF, num2word, words, poems
