import numpy as np
import chainer
from chainer import optimizers, Variable, serializers
import chainer.functions as F
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import argparse
import utils, model

parser = argparse.ArgumentParser(description='Snetiment Analysis with Stacked Gated Convolutional Neural Network')
parser.add_argument('-m', '-minimum_frequency', type=int, default=3)
parser.add_argument('-b', '-batch_size', type=int, default=500)
parser.add_argument('-e', '-epoch', type=int, default=200)
parser.add_argument('-s', '-embed_size', type=int, default=200)
parser.add_argument('-hs', '-hidden_size', type=int, default=-1)
parser.add_argument('-ss', '-stack_size', type=int, default=1)
parser.add_argument('-ml', '-minimum_sentence_length', type=int, default=-1)
parser.add_argument('-base', type=int, default=2)
parser.add_argument('-l', '-spatial_pyramid_pooling_level', type=int, default=2)
parser.add_argument('-o', '-output_size', type=int, default=2)
parser.add_argument('-w', '-kernel_width', type=int, default=2)
parser.add_argument('-a', '-activation', type=str, default='sigmoid')
parser.add_argument('-em', '-embed_matrix', type=open)
parser.add_argument('-padding', action='store_true')
parser.add_argument('-top_only', action='store_true')
parser.add_argument('-train_file', '-train_file', type=str, default='Tweet/train.txt')
parser.add_argument('-test_file', '-test_file', type=str, default='Tweet/test.txt')
args = parser.parse_args()

hidden_size = args.hs
if hidden_size < 0:
    hidden_size = args.s // (2 * args.base ** (args.ss))

ignore_sentence_length = args.ml
if args.padding:
    ignore_sentence_length = 0
if ignore_sentence_length < 0:
    ignore_sentence_length = args.w + 1

kernel_width = args.w
(vocab_size, train_data, train_label, test_data, test_label) = utils.read_data(args.train_file, args.test_file, args.m, ignore_sentence_length, kernel_width + (args.ss - 1))

train_pos = 0
train_neg = 0
for label in train_label:
    if label[0] == 0:
        train_neg += 1
    else:
        train_pos += 1
test_pos = 0
test_neg = 0
for label in test_label:
    if label == 0:
        test_neg += 1
    else:
        test_pos += 1
train_size = len(train_data)
print('Minimum_frequency: {0}\nIgnore_sentence_length: {1}\nTrain_size: {2}\nTest_size: {3}\nTrain_label: ({4}, {5})\nTest_label: ({6}, {7})\n'.format(args.m, ignore_sentence_length, train_size, len(test_data), train_pos, train_neg, test_pos, test_neg))

activation = F.sigmoid
if args.a == 'relu':
    activation = F.relu
print('Minibatch_size: {0}\nMaximum_epoch: {1}\nEmbeding_size: {2}\nHidden_layer_size: {3}\nOutput_layer_size: {4}\nActivation: {5}\nKernel_width: {6}\nStack_size: {7}\nSpatial_pyramid_pooling_level: {8}\nOnly_top_layer_spatial_pyramid_pooing: {9}\n'.format(args.b, args.e, args.s, hidden_size, args.o, args.a, kernel_width, args.ss, args.l, args.top_only))
if args.em == None:
    model = model.StackGatedCNN(vocab_size, args.s, hidden_size, args.o, kernel_width, args.ss, args.base, args.l, activation, None)
else:
    embed_matrix = utils.read_embed_matrix(args.em)
    model = model.StackGatedCNN(vocab_size, args.s, hidden_size, args.o, kernel_width, args.ss, args.base, args.l, activation, embed_matrix)
optimizer = optimizers.Adam()
optimizer.setup(model)

with chainer.no_backprop_mode():
    pred = model(Variable(np.array(test_data[0], dtype=np.int32)), args.top_only)
    for data in test_data[1:]:
        pred = F.concat((pred, model(Variable(np.array(data, dtype=np.int32)), args.top_only)), axis = 0)
    print(0, 'Accuracy', F.accuracy(pred, Variable(np.array(test_label, dtype=np.int32))).data)

for epoch in range(args.e):
    optimizer.new_epoch()
    indexes = np.random.permutation(train_size)
    sum_loss = 0
    loop = 0
    for i in range(0, train_size, args.b):
        loop += 1
        loss = 0
        model.zerograds()
        for index in indexes[i:i + args.b]:

            answer = model(Variable(np.array(train_data[index], dtype=np.int32)), args.top_only)
            t = Variable(np.array(train_label[index], dtype=np.int32))
            loss += F.softmax_cross_entropy(answer, t) / indexes[i:i + args.b].shape[0]
        print(epoch+1, min(train_size, i + args.b), loss.data)
        sum_loss += loss
        loss.backward()
        optimizer.update()
    print(epoch+1, 'Loss', sum_loss.data / loop)
    output_file = 'model-{0}.npz'.format(epoch+1)
    serializers.save_npz(output_file, model)
    with chainer.no_backprop_mode():
        pred = model(Variable(np.array(test_data[0], dtype=np.int32)), args.top_only)
        for data in test_data[1:]:
            pred = F.concat((pred, model(Variable(np.array(data, dtype=np.int32)), args.top_only)), axis = 0)
        print(epoch+1, 'Accuracy', F.accuracy(pred, np.array(test_label, dtype=np.int32)).data)

print('Final result report.')
pred = F.argmax(pred, axis=1).data
print(accuracy_score(test_label, pred))
print(classification_report(test_label, pred))
print(confusion_matrix(test_label, pred))
