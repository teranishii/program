
# -*- coding: utf-8 -*-

import re
import numpy as np
import chainer
from chainer import ChainList, optimizers, training,Variable
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import preprocess
import os

def sentence_block_embed(embed, x):
    batch, length = x.shape
    _, units = embed.W.shape
    e = embed(x.reshape((batch * length, )))
    assert(e.shape == (batch * length, units))
    e = F.transpose(F.stack(F.split_axis(e, batch, axis=0), axis=0), (0, 2, 1))
    assert(e.shape == (batch, units, length))
    return e

# モデルクラスの定義
class CNN_SentenceClassifier(ChainList):
    def __init__(self, in_channel, out_channel, filter_height_list, filter_width, out_size, max_sentence_size):
        # クラスの初期化
        # :param in_channel: 入力チャネル数
        # :param out_channel: 出力チャネル数
        # :param filter_height_list: フィルター縦サイズの配列
        # :param filter_width: フィルター横サイズ
        # :param out_size: 分類ラベル数
        # :param max_sentence_size: 文章の長さの最大サイズ
        self.filter_height_list = filter_height_list
        self.max_sentence_size = max_sentence_size
        self.convolution_num = len(filter_height_list)
        self.embed_x = L.EmbedID(filter_width,200,ignore_label=-1)
        # Linkの定義
        link_list = [L.Convolution2D(in_channel, out_channel, (i, 200), pad=0) for i in filter_height_list] # Convolution層用のLinkをフィルター毎に追加
        link_list.append(L.Linear(out_channel * self.convolution_num, out_channel * self.convolution_num)) # 隠れ層
        link_list.append(L.Linear(out_channel * self.convolution_num, out_size)) # 出力層
        """
        self.gcnn = []
        for i in range(stack):
            self.gcnn.append(Conv_Gate(embedding_size , (embedding_size, kernel_width)))
        """  
        
        # 定義したLinkのリストを用いてクラスを初期化する
        super(CNN_SentenceClassifier, self).__init__(*link_list)
 
    def __call__(self, x):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        # フィルタを通した結果を格納する配列

        exs=self.embed_x(Variable(np.array(x,dtype=np.int32)))
        exs=F.expand_dims(exs,axis=1)
        
        xcs = [None for i in self.filter_height_list]
        chs = [None for i in self.filter_height_list]
        
        exs=self.embed_x(Variable(np.array(x,dtype=np.int32)))
        exs=F.expand_dims(exs,axis=1)
        
        # フィルタごとにループ
                
        for i, filter_height in enumerate(self.filter_height_list):
            xcs[i] = F.relu(self[i](exs))
            chs[i] = F.max_pooling_2d(xcs[i], (self.max_sentence_size+1-filter_height))
        # Convolution+Poolingの結果の結合
        h = F.concat(chs, axis=2)
        h = F.dropout(F.tanh(self[self.convolution_num+0](h)))
        y = self[self.convolution_num+1](h)
        return y
 
# 学習
en_path = os.path.join("./", "train/body.txt")
source_vocab = ['<eos>', '<unk>', '<bos>'] + preprocess.count_words(en_path, 900)
source_data = preprocess.make_dataset(en_path, source_vocab)
source_ids = {word: index for index, word in enumerate(source_vocab)}
words = {i: w for w, i in source_ids.items()}
N = len(source_data)

words[len(words)]="padding"

a=[0]*(1000)
b=[1]*(1000)
data_t=a+b

max_len=0
for k in range(len(source_data)):
    if max_len<len(source_data[k]):
        max_len=len(source_data[k])
for k in range(len(source_data)):
    source_data[k]=source_data[k][:max_len] + [-1]*(max_len-len(source_data[k]))
"""
a_on = np.identity(len(words))[source_data]
a_on=np.array(a_on,dtype="float32")
a_on=a_on[:,:,0:len(words)-1]

dataset = []
for x, t in zip(a_on, data_t):
    dataset.append((x.reshape(1, max_len, len(words)-1), t))
"""
a_on=source_data
a_on=np.array(a_on,dtype="float32")
dataset = []
for x, t in zip(a_on, data_t):
    dataset.append((x, t))


# 定数
EPOCH_NUM = 50
BATCH_SIZE = 100
OUT_SIZE = 2
FILTER_HEIGHT_LIST = [1,2,3]
OUT_CHANNEL = 16
max_sentence_size = max_len

# モデルの定義
model = L.Classifier(CNN_SentenceClassifier(
    in_channel=1,
    out_channel=OUT_CHANNEL,
    filter_height_list=FILTER_HEIGHT_LIST,
    filter_width=len(words)-1,
    out_size=OUT_SIZE,
    max_sentence_size=max_sentence_size))
optimizer = optimizers.Adam()
optimizer.setup(model)

# 学習開始
train, test = chainer.datasets.split_dataset_random(dataset, N-200)
train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(test, 200, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result-cnn")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"])) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
trainer.extend(extensions.PlotReport(['main/loss','validation/main/loss'] ,x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/accuracy'] ,x_key='epoch', file_name='accuracy.png'))
#trainer.extend(extensions.ProgressBar()) # プログレスバー出力
trainer.run()