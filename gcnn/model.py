import numpy as np
from chainer import Chain, ChainList, Variable
import chainer.links as L
import chainer.functions as F

import time

class Conv_Gate(ChainList):
    def __init__(self, nkernel, kernel_size):
        super(Conv_Gate, self).__init__(
            L.Convolution2D(None, nkernel, kernel_size),
            L.Convolution2D(None, nkernel, kernel_size),
        )
        self.channel = nkernel

    def __call__(self, x):
         h1 = self[0](x)
         h2 = F.sigmoid(self[1](x))
         return (h1 * h2).reshape(1, 1, self.channel, -1)

class StackGatedCNN(Chain):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, kernel_width, stack, base, level, activation, embed_matrix):
        super(StackGatedCNN, self).__init__()
        self.pre_train = False
        with self.init_scope():
            self.embedID = L.EmbedID(vocab_size, embedding_size, ignore_label=0)
            if not embed_matrix == None:
                self.embedID = L.EmbedID(vocab_size, embedding_size, embed_matrix, ignore_label=0)
                self.pre_train = True
            self.gcnn = []
            for i in range(stack):
                self.gcnn.append(Conv_Gate(embedding_size // (base ** (i + 1)), (embedding_size//(2 ** i), kernel_width)))
            self.pooling = []
            for i in range(level):
                for j in range(2 ** i):
                    self.pooling.append(L.Linear(embedding_size//(base ** stack), hidden_size))
            self.clf = L.Linear(hidden_size, output_size)
        self.gcnn_stack = stack
        self.spp_level = level
        self.activation = activation

    def __call__(self, x, flag):
        
        h = F.transpose(self.embedID(x), axes=(0, 1, 3, 2))
        
        if self.pre_train:
            h.unchain_backward()
        
        for i in range(self.gcnn_stack):
            h = self.gcnn[i](h)
        h = self.spartial_pyramid_pooling(h)
        start = 0
        if flag:
            for i in range(self.spp_level-1):
                start += 2 ** i
        tmp = 0
        for i in range(start, len(h)):
            tmp += self.pooling[i](h[i])
        h = self.activation(tmp)
        return F.softmax(self.clf(h))

    def spartial_pyramid_pooling(self, x):
        padding = Variable(np.zeros((1, x.shape[2]), dtype=np.float32))
        h = [F.expand_dims(F.flatten(F.max(x, axis=3)), axis=0)]
        length = x.shape[3]
        for i in range(1, self.spp_level):
            division = 2 ** i
            window_size = length // division
            if window_size > 0:
                for j in range(i):
                    h.append(F.expand_dims(F.flatten(F.max(x[:,:,:,(window_size * j):(window_size * (j + 1))], axis=3)), axis=0))
                h.append(F.expand_dims(F.flatten(F.max(x[:,:,:,(window_size * i):], axis=3)), axis=0))
            else:
                for j in range(length):
                    h.append(F.expand_dims(F.flatten(x[:,:,:,j]), axis=0))
                extend = division - length     
                for j in range(extend):
                    h.append(padding)
        return(h)


           
            
        
