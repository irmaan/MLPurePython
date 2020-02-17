from model import LSTM
import tensorflow as tf
import numpy as np
from random import shuffle

train_input = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(train_input)
train_input = [map(int, i) for i in train_input]
ti = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []

for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)

NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:] #everything beyond 10,000

train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES] #till 10,000
 
batch_size = 1000
no_of_batches = int(len(train_input)/batch_size)
epoch = 100

cell = LSTM(batch_size, 24)

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
        ptr+=batch_size
        sess.run(cell.train, {cell.data: inp, cell.target: out})
    print "Epoch - ",str(i)
incorrect = sess.run(cell.error, {cell.data: test_input, cell.target: test_output})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
