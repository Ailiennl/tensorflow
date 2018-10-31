#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import tempfile

Mnist=input_data.read_data_sets('./',one_hot=True)

def conv(x, filter_):
    return tf.nn.conv2d(x, filter_, strides=[1, 1, 1, 1], padding='SAME')


def pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init, name='weight')


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init, name='bias')

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])


with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
    filter_1 = weight_variable([3, 3, 1, 32])
    b1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv(x_image, filter_1) + b1)

with tf.name_scope('pool1'):
    h_pool1 = pool(h_conv1)

with tf.name_scope('conv2'):
    filter_2 = weight_variable([5, 5, 32, 64])
    b2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv(h_pool1, filter_2) + b2)

with tf.name_scope('pool2'):
    h_pool2 = pool(h_conv2)

with tf.name_scope('fc1'):
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h2_flat = tf.reshape(h_pool2, [-1,7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h2_flat, w_fc1)+b_fc1)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_pre = tf.matmul(h_fc1_drop, w_fc2) + b_fc2


cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_pre))
optimizer =tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pre=tf.cast(tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1)),tf.float32)
accuracy=tf.reduce_mean(correct_pre)
graph_location=tempfile.mkdtemp()
print('Saving graph to:',graph_location)
train_writer=tf.summary.FileWriter(graph_location,tf.get_default_graph())

init=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    train_epoch=[]
    train_acc=[]
    test_epoch = []
    test_acc = []
    for i in range(10001):
        batch=Mnist.train.next_batch(50)
        # print(type(batch[0]),type(batch[1]))
        sess.run(optimizer,feed_dict={x:batch[0],y:batch[1],keep_prob:0.5})
        if i%200==0:
            train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y:batch[1],keep_prob:1})
            train_epoch.append(i)
            train_acc.append(train_accuracy)
            print('step:{} train accuracy:{}'.format(i,train_accuracy))
            test_batch = Mnist.test.next_batch(1000)
            test_accuracy = sess.run(accuracy, feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1})
            print('step:{} test accuracy:{}'.format(i, test_accuracy))
            test_epoch.append(i)
            test_acc.append(test_accuracy)
    save_path=saver.save(sess, './mnist_cnn_model.ckpt')
    train_epoch=np.array(train_epoch)
    train_acc=np.array(train_acc)
    np.save('train_epoch.npy',train_epoch)
    np.save('train_acc.npy',train_acc)
    test_epoch=np.array(test_epoch)
    test_acc=np.array(test_acc)
    np.save('test_epoch.npy',test_epoch)
    np.save('test_acc.npy',test_acc)

    plt.figure(1)
    plt.plot(train_epoch, train_acc)
    plt.title('train_acc')
    plt.savefig('train_acc.jpg')
    plt.figure(2)
    plt.plot(test_epoch, test_acc)
    plt.title('test_acc')
    plt.savefig('test_acc.jpg')
