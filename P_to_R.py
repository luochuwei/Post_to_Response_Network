#-*- coding:utf-8 -*-
###############################################
#
#    Author: Chuwei Luo(changing lipiji's code)
#    Email: luochuwei@gmail.com
#    Date: 11/02/2016
#    Usage: Main
#
###############################################
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *
import data

# use_gpu(1) # -1:cpu; 0,1,2,..: gpu

e = 0.01   #error
lr = 0.5
drop_rate = 0.
batch_size = 1 #To run this code, data_size mod batch_size must equal 0
hidden_size = [500]
# try: gru, lstm
cell = "gru"
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam
optimizer = "adadelta" 

seqs, i2w, w2i, data_xy = data.word_sequence("data/toy2.txt", batch_size)
seqs2, i2w2, w2i2, data_xy2 = data.word_sequence("data/toy3.txt", batch_size)
dim_x = len(w2i)
dim_y = len(w2i2)
num_sents = data_xy[0][3]
for i in data_xy:
    data_xy[i][1] = data_xy2[i][0]
    data_xy[i].append(data_xy2[i][-2])
print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = RNN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate, num_sents)

print "training..."
# start = time.time()
# g_error = 9999.9999
# for i in xrange(2000):
#     error = 0.0
#     in_start = time.time()
#     for batch_id, xy in data_xy.items():
#         X = xy[0]
#         Y = 
#         mask = xy[2]
#         local_batch_size = xy[3]
#         cost, sents, test, test2, test3, test4, test5,test6,test7,test8 = model.train(X, Y, mask, mask_y, lr, local_batch_size)
#         error += cost
#         break
#         #print i, g_error, (batch_id + 1), "/", len(data_xy), cost
#     in_time = time.time() - in_start
#     break

#     #打印结果
#     for s in xrange(int(sents.shape[1] / dim_y)):
#         xs = sents[:, s * dim_y : (s + 1) * dim_y]
#         for w_i in xrange(xs.shape[0]):
#             w = i2w[np.argmax(xs[w_i, :])]
#             if w == "<eoss>":
#                 break
#             print w,
#         print "\n"

#     error /= len(data_xy);
#     if error < g_error:
#         g_error = error

#     print "Iter = " + str(i) + ", Error = " + str(error) + ", Time = " + str(in_time)
#     if error <= e:
#         break

# print "Finished. Time = " + str(time.time() - start)

# print "save model..."
# save_model("./model/hed.model", model)

start = time.time()
g_error = 9999.9999
for i in xrange(2000):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_xy.items():
        X = xy[0]
        Y = xy[1]
        mask = xy[2]
        local_batch_size = xy[3]
        mask_y = xy[-1]
        cost, sents, test, test2, test3, test4, test5,test6,test7,test8 = model.train(X, Y, mask, mask_y, lr, local_batch_size)
        error += cost
        # break
        #print i, g_error, (batch_id + 1), "/", len(data_xy), cost
    in_time = time.time() - in_start
    # break

    #打印结果
    for s in xrange(int(sents.shape[1] / dim_y)):
        xs = sents[:, s * dim_y : (s + 1) * dim_y]
        for w_i in xrange(xs.shape[0]):
            w = i2w2[np.argmax(xs[w_i, :])]
            if w == "<eoss>":
                break
            print w,
        print "\n"

    error /= len(data_xy);
    if error < g_error:
        g_error = error

    print "Iter = " + str(i) + ", Error = " + str(error) + ", Time = " + str(in_time)
    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/hed.model", model)
