#-*- encoding:utf-8 -*-
#!/usr/local/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def add_layer(inputs, in_size, out_size, activation_function=None):
    w = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros(out_size) + 0.01)

    z = tf.matmul(inputs, w) + b

    if activation_function == None:
        output = z
    else:
        output = activation_function(z)

    return output


if __name__ == '__main__':
    Mnist = input_data.read_data_sets('./', one_hot=True)
    learning_rate = 0.01
    n_epoch = 10
    batch_size = 128
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    l1 = add_layer(x, 784, 500, activation_function=tf.nn.relu)
    l2 = add_layer(l1, 500, 200, activation_function=tf.nn.relu)
    l_pre = add_layer(l2, 200, 10)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=l_pre))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        n_batchs = int(Mnist.train.num_examples / batch_size)
        for i in range(n_epoch):
            for j in range(n_batchs):
                x_train, y_train = Mnist.train.next_batch(batch_size)
                _, loss_ = sess.run([optimizer, loss], feed_dict={x: x_train, y: y_train})
                if j == 0:
                    print("Loss of epochs[{0}]: {1}".format(i, loss_))

        n_batchs = int(Mnist.test.num_examples / batch_size)
        total = 0
        for j in range(n_batchs):
            x_test, y_test = Mnist.test.next_batch(batch_size)
            pre = sess.run(l_pre, feed_dict={x: x_test})
            pre = tf.nn.softmax(pre)
            correct_p = tf.equal(tf.argmax(pre, 1), tf.argmax(y_test, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_p, tf.float32))
            total += sess.run(accuracy)

        print(total / Mnist.test.num_examples)

