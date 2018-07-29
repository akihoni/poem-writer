# poem-writer
在config.py中修改配置，在有模型的情况下在终端中运行predict.py可以直接自动作诗。

训练集有两个，两个都是五言诗，区别不大...但可以在config.py中修改要使用的训练集

网络结构为Embedding+LSTM+Dropout+LSTM+Dropout+Dense，之后还会尝试继续优化。

自己训练的模型过大传不上来...且藏头诗的输出非常不科学，算法还需要优化。
