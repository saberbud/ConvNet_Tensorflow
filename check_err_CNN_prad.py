from __future__ import division, print_function, absolute_import
import tensorflow as tf
import math
import numpy as np
from tensorflow.python.framework import ops

# Import xydata
from input_xy_data import xy_data
xyd=xy_data()
xyd.read_in(5)

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
    return X, Y, keep_prob


def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [5, 5, 2, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def forward_propagation(X, parameters, dropout):
    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
    P2f = tf.contrib.layers.flatten(P2)

    Z3 = tf.contrib.layers.fully_connected(P2f, 1024, activation_fn=None)
    Z3d = tf.nn.dropout(Z3, dropout)

    Zout=tf.contrib.layers.fully_connected(Z3d, 2, activation_fn=None)

    return Zout


def compute_cost(Zout, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Zout, labels=Y))
    return cost


def model(num_steps, batch_size=128, learning_rate=0.001, display_step=10):
    (n_H0, n_W0, n_C0) = (36,36,2)
    n_y = 2

    xyd.shape(n_H0)
    xyd.make_batch(batch_size)

    X, Y, keep_prob = create_placeholders(n_H0, n_W0, n_C0, n_y)

    lrd=tf.placeholder(tf.float32)  #learning rate decay

    parameters = initialize_parameters()

    Zout = forward_propagation(X, parameters, keep_prob)

    cost = compute_cost(Zout, Y)

    predict_op = tf.argmax(Zout, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    optimizer = tf.train.AdamOptimizer(learning_rate=lrd).minimize(cost)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for step in range(1, num_steps+1):
            batch_x, batch_y = xyd.xy_next_batch()

            if step > 200:
                learning_rate=0.0002

            if step > 300:
                learning_rate=0.00005

            sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y, keep_prob: 0.8, lrd:learning_rate})

            if step % display_step == 0 or step == 1:
                tempcost, acc= sess.run([cost, accuracy], feed_dict={X:batch_x, Y:batch_y, keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch cost= " + \
                      "{:.4f}".format(tempcost) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc) + ", learning rate= "+ \
                      "{:.5f}".format(learning_rate))

        print("Optimization Finished.")

        Xt, Yt= xyd.out_test_good(0, -1)
        test_good_acc=sess.run(accuracy, feed_dict={X: Xt,
                                               Y: Yt,
                                               keep_prob: 1.0})

        print("Testing Good Accuracy:" + str(test_good_acc))

        Xt, Yt= xyd.out_test_bad(0, -1)
        test_bad_acc=sess.run(accuracy, feed_dict={X: Xt,
                                               Y: Yt,
                                               keep_prob: 1.0})

        print("Testing Bad Accuracy:" + str(test_bad_acc))

        save_path=saver.save(sess, "/Users/saberbud/Programs/CNN_prad/xy_CNN_prad.ckpt")
        print("Saved: " + str(save_path))

    return parameters


def Test_accuracy(parameters):
    (n_H0, n_W0, n_C0) = (36,36,2)
    n_y = 2

    xyd.shape(n_H0)

    X, Y, keep_prob = create_placeholders(n_H0, n_W0, n_C0, n_y)

    Zout = forward_propagation(X, parameters, keep_prob)

    predict_op = tf.argmax(Zout, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "/Users/saberbud/Programs/CNN_prad/xy_CNN_prad.ckpt")

        Xt, Yt= xyd.out_test_good(0, -1)
        test_good_acc=sess.run(accuracy, feed_dict={X: Xt,
                                               Y: Yt,
                                               keep_prob: 1.0})


        Xt, Yt= xyd.out_test_bad(0, -1)
        test_bad_acc=sess.run(accuracy, feed_dict={X: Xt,
                                               Y: Yt,
                                               keep_prob: 1.0})


    return test_good_acc, test_bad_acc



#Training
#with tf.variable_scope("model"):
#    parameters=model(400, batch_size=128, learning_rate=0.001, display_step=10)

#print(parameters)
#print("Global variables:")
#print(tf.global_variables())


print("Calculate test accuracy in function:")
with tf.variable_scope("model"):
    parameters = initialize_parameters()
    test_good_acc,test_bad_acc=Test_accuracy(parameters)
    print("test good accuracy= " + str(test_good_acc))
    print("test bad accuracy= " + str(test_bad_acc))















