#!/usr/bin/python
# -*- codding:utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

Mnist = input_data.read_data_sets("./", one_hot=True)

learning_rate = 0.01
batch_size = 128
n_epoch = 25

x = tf.placeholder(tf.float32, [batch_size, 784])
y = tf.placeholder(tf.float32, [batch_size, 10])

w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.1), name='weight')
b = tf.Variable(tf.zeros([1, 10]), name='bias')

logits = tf.matmul(x, w) + b

entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batchs = int(Mnist.train.num_examples / batch_size)
    for i in range(n_epoch):
        for j in range(n_batchs):
            x_batch, y_batch = Mnist.train.next_batch(batch_size)
            _, loss_ = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch})
            # if j%100==0:
            #     print("Loss of epochs[{0}] batch[{1}]: {2}".format(i, j, loss_))

    t_batchs = int(Mnist.test.num_examples / batch_size)
    total_correct_pre = 0
    for i in range(t_batchs):
        x_batch, y_batch = Mnist.test.next_batch(batch_size)
        pred = sess.run(logits, feed_dict={x: x_batch})
        correct_pre = tf.equal(tf.argmax(pred, 1), tf.argmax(y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_pre, tf.float32))

        total_correct_pre += sess.run(accuracy)

    print('Accuracy:{}'.format(total_correct_pre / Mnist.test.num_examples))








