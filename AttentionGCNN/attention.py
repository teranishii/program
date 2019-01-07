# -*- coding: utf-8 -*-
#!/usr/bin/env python

import argparse
import datetime
import os
import sys
import six
import numpy

import chainer
from chainer import cuda ,Variable,serializers,ChainList
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.dataset import convert

import preprocess

EOS=0
UNK=1
BOS=2

WEIGHT=True
TRANS_ALL=False

if not os.path.isdir("./weight"):
    os.mkdir("./weight")
if not os.path.isdir("./result"):
    os.mkdir("./result")
if not os.path.isdir("./summary"):
    os.mkdir("./summary")

def seq2seq_pad_concat_convert(xy_batch, device, eos_id=0, bos_id=2):
    x_seqs, y_seqs = zip(*xy_batch)
    
    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    y_block = convert.concat_examples(y_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)

    y_out_block = xp.pad(y_block, ((0, 0), (0, 1)),
                         'constant', constant_values=-1)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    y_in_block = xp.pad(y_block, ((0, 0), (1, 0)),
                        'constant', constant_values=bos_id)
    return (x_block, y_in_block, y_out_block)

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs

def sentence_block_embed(embed, x):
    batch, length = x.shape
    _, units = embed.W.shape
    e = embed(x.reshape((batch * length, )))
    assert(e.shape == (batch * length, units))
    e = F.transpose(F.stack(F.split_axis(e, batch, axis=0), axis=0), (0, 2, 1))
    assert(e.shape == (batch, units, length))
    return e

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
         return F.transpose((h1 * h2),(0,2,1,3))

class Att(chainer.Chain):    
    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,batch,weight,stack,width):
        super(Att, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units,ignore_label=-1)
            self.embed_y = L.EmbedID(n_target_vocab, n_units,ignore_label=-1)
            self.embed_yy = L.EmbedID(n_target_vocab, n_units,ignore_label=-1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0)
            self.decoder2 = L.NStepLSTM(n_layers, n_units, n_units, 0)
            self.Ws = L.Linear(n_units, n_target_vocab)
            self.We = L.Linear(n_units, n_target_vocab)
            self.V = L.Linear(n_units, n_target_vocab)
            self.gcnn = []
            for i in range(stack):
                self.gcnn.append(Conv_Gate(n_units,(n_units,width)))
                
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout = 0.1
        self.scale_score=1. / (n_units)**0.5
        self.batch=batch
        self.weight=weight
        self.stack=stack
        self.width=width
        
    def make_input_embedding(self, embed, block):
        batch, length = block.shape
        #embed層の転置(batch,hidden_layer,word_len)->(batch,word_len,hidden_layer)
        emb_block = sentence_block_embed(embed, block)
        #emb_block = F.dropout(emb_block, self.dropout)
        return emb_block
    
    def __call__(self, x_block, y_in_block, y_out_block):
        
        batch = len(x_block)
        #embed
        ex_block = F.dropout(self.make_input_embedding(self.embed_x, x_block), self.dropout)
        ey_block = F.dropout(self.make_input_embedding(self.embed_y, y_in_block), self.dropout)
        eyy_block = F.dropout(self.make_input_embedding(self.embed_yy, y_in_block), self.dropout)
        eys = F.transpose(ey_block,(0, 2, 1))
        eyys = F.transpose(eyy_block,(0, 2, 1))
        #gcnn
        h=F.expand_dims(ex_block,axis=1)
        for i in range(self.stack):
            h = self.gcnn[i](h)
        h=F.dropout(F.squeeze(h,axis=1),self.dropout)
        #Nsteolstm
        eys2=[i for i in eys]
        eyys2=[i for i in eyys]
        _, _, oss = self.decoder(None, None, eys2)
        _, _, oss2 = self.decoder2(None, None, eyys2)
        ss=F.stack(oss,axis=0)
        ss2=F.stack(oss2,axis=0)
        #mask_make
        mask = (y_in_block[:, :,None] >= 0)*self.xp.ones((self.batch,1,self.n_units), dtype=bool)
        ss = F.where(mask, ss, self.xp.full(ss.shape, 0, 'f'))
        #weight_calclate
        batch_A = F.batch_matmul(ss,h)* self.scale_score
        mask = (x_block[:,0:len(x_block[0])-self.stack*(self.width-1)][:, None,:] >= 0)*(y_in_block[:, :,None] >= 0)
        batch_A = F.where(mask, batch_A, self.xp.full(batch_A.shape, -self.xp.inf, 'f'))
        batch_A = F.softmax(batch_A, axis=2)
        batch_A = F.where(self.xp.isnan(batch_A.data), self.xp.zeros(batch_A.shape, 'f'), batch_A)
        batch_A, h = F.broadcast(batch_A[:, None], h[:, :, None])        
        batch_C = F.sum(batch_A * h, axis=3)
        
        e = F.transpose(batch_C,(0, 2, 1))
        e = F.squeeze(F.concat(F.split_axis(e, self.batch, axis=0), axis=1))
        ss2 = F.squeeze(F.concat(F.split_axis(ss2, self.batch, axis=0), axis=1))
        t=(self.We(e)+self.Ws(ss2))
        t= F.dropout(t, self.dropout)
        
        concat_ys_out = F.concat(y_out_block, axis=0)
        loss = F.sum(F.softmax_cross_entropy(t, concat_ys_out, reduce='no')) / batch
        
        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss

    def translate(self, xs, max_length=100):
        xs=numpy.insert(xs,0,2)
        xs=numpy.append(xs,0)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            exs=self.embed_x(Variable(self.xp.array(xs,dtype=self.xp.int32)))
            
            h=F.expand_dims(exs,axis=0)
            h=F.expand_dims(h,axis=0)
            h=F.transpose(h,(0,1,3,2))
            for i in range(self.stack):
                h = self.gcnn[i](h)
            h=F.squeeze(h,axis=1)
            h=F.squeeze(h,axis=0)
            h=F.transpose(h,(1, 0))
            
            ys = self.xp.full(1,2, self.xp.int32)
            result = []
            hx=None
            cx=None
            hx2=None
            cx2=None

            for i in range(max_length):
                eys = self.embed_y(ys)
                eyys = self.embed_yy(ys)
                eys2=[eys]
                eyys2=[eyys]
                hx, cx, ss = self.decoder(hx, cx, eys2)
                hx2, cx2, ss2 = self.decoder2(hx2, cx2, eyys2)
                
                batch_A = F.matmul(h,ss[0],transb=True)* self.scale_score
                batch_A = F.softmax(batch_A,axis=0)
                if self.weight:
                    with open("weight/wei.txt","a",encoding="utf-8") as f:
                        for j in range(len(batch_A)):
                            f.write(str(batch_A[j][0].data)+"\n")
                        f.write("--------------\n")
                s=F.matmul(batch_A,h,transa=True)
                t=(self.We(s)+self.Ws(ss2[0]))
                ys = self.xp.argmax(t.data, axis=1).astype(self.xp.int32)
                if ys[0]==0:
                    break
                result.append(ys)
        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)
        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total

def main():
    parser = argparse.ArgumentParser(description='Chainer example: Att_summary')
    parser.add_argument('--source', '-s' , type=str,
                        default='test/body.txt',
                        help='source sentence list')
    parser.add_argument('--target', '-t' , type=str,
                        default='test/sum.txt',
                        help='target sentence list')
    parser.add_argument('--source_valid',type=str,
                        default='test/body.txt',
                        help='source sentence list for validation')
    parser.add_argument('--target_valid',type=str,
                        default='test/sum.txt',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=200,
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='number of iteration to show log')
    parser.add_argument('--validation_interval', type=int, default=20,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    args = parser.parse_args()

    # Load pre-processed dataset
    print('[{}] Loading dataset... (this may take several minutes)'.format(datetime.datetime.now()))
    
    en_path = os.path.join("./", args.source)
    #引数は語彙数
    source_vocab = ['<eos>', '<unk>', '<bos>'] + preprocess.count_words(en_path, 18000)
    source_data = preprocess.make_dataset(en_path, source_vocab)
    fr_path = os.path.join("./", args.target)
    target_vocab = ['<eos>', '<unk>', '<bos>'] + preprocess.count_words(fr_path, 18000)
    target_data = preprocess.make_dataset(fr_path, target_vocab)
    assert len(source_data) == len(target_data)
    print('Original training data size: %d' % len(source_data))
    train_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)]
    print('Filtered training data size: %d' % len(train_data))

    source_ids = {word: index for index, word in enumerate(source_vocab)}
    target_ids = {word: index for index, word in enumerate(target_vocab)}
    #{}の中身を入れ換え
    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}
    
    # Setup model
    stack=1
    width=3
    print("aaa")
    model = Att(args.layer, len(source_ids), len(target_ids), args.unit,args.batchsize, WEIGHT,stack,width)
    print("aaa")
    if args.gpu >= 0:
        print("aaa")
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
    # Setup optimizer
    print("aaa")
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    # Setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    # Setup updater and trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=seq2seq_pad_concat_convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'elapsed_time']),
        trigger=(1, 'epoch'))
    
    if args.source_valid and args.target_valid:
        en_path = os.path.join("./", args.source_valid)
        source_data = preprocess.make_dataset(en_path, source_vocab)
        fr_path = os.path.join("./", args.target_valid)
        target_data = preprocess.make_dataset(fr_path, target_vocab)
        assert len(source_data) == len(target_data)
        test_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)]
        test_source_unknown = calculate_unknown_ratio(
            [s for s, _ in test_data])
        test_target_unknown = calculate_unknown_ratio(
            [t for _, t in test_data])

        print('Validation data: %d' % len(test_data))
        print('Validation source unknown ratio: %.2f%%' % (test_source_unknown * 100))
        print('Validation target unknown ratio: %.2f%%' % (test_target_unknown * 100))

        @chainer.training.make_extension()
        def translate_one(trainer):
            #訓練文での生成の場合 defaltは0番目
            
            a, b = map(list, zip(*train_data))
            source=a[21]
            target=b[21]
            result = model.translate(source)[0]
            """
            #テストランダムの場合
            source, target = test_data[numpy.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]
            """
            source_sentence = ' '.join([source_words[x] for x in source])
            target_sentence = ' '.join([target_words[y] for y in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            print('# source : ' + source_sentence)
            print('# result : ' + result_sentence)
            print('# expect : ' + target_sentence)
            
            if WEIGHT:
                with open("weight/wei.txt","a",encoding="utf-8") as f:
                    f.write("<body> <fos> "+str(source_sentence)+"\n")
                    f.write("<generation> <fos> "+str(result_sentence)+"\n")
        
        def translate_all(trainer):
            a, b = map(list, zip(*test_data))
            for k in range(len(test_data)):
                source=a[k]
                result = model.translate(source)[0]
                result_sentence = ' '.join([target_words[y] for y in result])
                with open("summary/result.txt","a",encoding="utf-8") as f:
                    f.write(str(result_sentence)+"\n")
            sys.exit(0)
        """
        if TRANS_ALL:
            trainer.extend(translate_all, trigger=(19, 'epoch'))
        """
        trainer.extend(translate_one, trigger=(args.validation_interval, 'epoch'))
        if TRANS_ALL:
            trainer.extend(translate_all, trigger=(20, 'epoch'))
        test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, False, False)
        trainer.extend(extensions.Evaluator(
            test_iter, model, converter=seq2seq_pad_concat_convert,device=args.gpu))
        trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
        #trainer.extend(extensions.PlotReport(
        #    ['main/loss','validation/main/loss'] ,x_key='epoch', file_name='loss.png'))
    print('start training')
    """
    #save_model_load
    filename = "./result/snapshot_iter_779"
    serializers.load_npz(filename, trainer)   
    """
    trainer.run()

if __name__ == '__main__':
    main()
