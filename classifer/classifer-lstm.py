# -*- coding: utf-8 -*-

import re
import numpy as np
import chainer
from chainer import Chain, optimizers, training,Variable,serializers
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
class LSTM_SentenceClassifier(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size):
        # クラスの初期化
        # :param vocab_size: 単語数
        # :param embed_size: 埋め込みベクトルサイズ
        # :param hidden_size: 隠れ層サイズ
        # :param out_size: 出力層サイズ
        super(LSTM_SentenceClassifier, self).__init__(
            # encode用のLink関数
            xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = L.LSTM(embed_size, hidden_size),
            encoder = L.NStepLSTM(1, embed_size, hidden_size, 0.1),
            hh = L.Linear(hidden_size, hidden_size),
            # classifierのLink関数
            hy = L.Linear(hidden_size, out_size)
        )
    
    def make_input_embedding(self, embed, block):
        batch, length = block.shape
        #embed層の転置をとってるだけ(batch,hidden_layer,word_len)->(batch,word_len,hidden_layer)
        emb_block = sentence_block_embed(embed, block)
        #emb_block = F.dropout(emb_block, self.dropout)
        return emb_block
 
    def __call__(self, x):
        # 順伝播の計算を行う関数
        # :param x:　入力値
        # エンコード
        """
        ex_block = self.make_input_embedding(self.xe, x)
        ex_block = F.dropout(ex_block, 0.3)
        exs = F.transpose(ex_block,(0, 2, 1))
        exs2=[i for i in exs]
        h, _, _ = self.encoder(None, None, exs2)
        
        """
        x = F.transpose_sequence(x)
        self.eh.reset_state()
        for word in x:
            e = self.xe(word)
            e=F.dropout(e,ratio=0.1)
            h = self.eh(e)
            h=F.dropout(h,ratio=0.1)
        
        y = self.hy(h)
        return y
 
# 学習
en_path = os.path.join("./", "train/sum.txt")
source_vocab = ['<eos>', '<unk>', '<bos>'] + preprocess.count_words(en_path, 10000)
source_data = preprocess.make_dataset(en_path, source_vocab)
source_ids = {word: index for index, word in enumerate(source_vocab)}
words = {i: w for w, i in source_ids.items()}
N = len(source_data)

a=[0]*(1000)
b=[1]*(1000)
data_t=a+b

max_len=0
for k in range(len(source_data)):
    if max_len<len(source_data[k]):
        max_len=len(source_data[k])
for k in range(len(source_data)):
    source_data[k]=source_data[k][:max_len] + [-1]*(max_len-len(source_data[k]))
data_x_vec=source_data
dataset = []
for x, t in zip(data_x_vec, data_t):
    dataset.append((x, t))

# 定数
EPOCH_NUM = 200
EMBED_SIZE = 200
HIDDEN_SIZE = 200
BATCH_SIZE = 100
OUT_SIZE = 2

# モデルの定義
model = L.Classifier(LSTM_SentenceClassifier(
    vocab_size=len(words),
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    out_size=OUT_SIZE
))
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# 学習開始
train, test = chainer.datasets.split_dataset_random(dataset, N-200)
train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(test, 200, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"])) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
trainer.extend(extensions.PlotReport(['main/loss','validation/main/loss'] ,x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/accuracy'] ,x_key='epoch', file_name='accuracy.png'))
#trainer.extend(extensions.ProgressBar()) # プログレスバー出力
trainer.run()