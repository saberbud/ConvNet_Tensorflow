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

    Wfc1 = tf.get_variable("Wfc1", [5184, 1024], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    bfc1 = tf.get_variable("bfc1", [1024], initializer=tf.zeros_initializer())

    Wfc2 = tf.get_variable("Wfc2", [1024, 2], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    bfc2 = tf.get_variable("bfc2", [2], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "W2": W2,
                  "Wfc1": Wfc1,
                  "bfc1": bfc1,
                  "Wfc2": Wfc2,
                  "bfc2": bfc2}

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

    #Z3 = tf.contrib.layers.fully_connected(P2f, 1024, activation_fn=None)
    Wfc1=parameters['Wfc1']
    bfc1=parameters['bfc1']
    Z3 = tf.add(tf.matmul(P2f, Wfc1),bfc1)
    Z3d = tf.nn.dropout(Z3, dropout)

    #Zout=tf.contrib.layers.fully_connected(Z3d, 2, activation_fn=None)
    Wfc2=parameters['Wfc2']
    bfc2=parameters['bfc2']
    Zout = tf.add(tf.matmul(Z3d, Wfc2),bfc2)

    #return [Z1, A1, P1, Z2, A2, P2, P2f, Z3, Z3d, Zout]
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


'''
Xin, X, Y, keep_prob= create_placeholders(28, 28, 1, 10)
print("Xin = " + str(Xin))
print("X = " + str(X))
print("Y = " + str(Y))
print("keep_prob=" + str(keep_prob))
'''

'''
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = " + str(parameters["W1"].eval()[1,1,0]))
    print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
    print("W1 shape= " +  str(parameters["W1"].get_shape()))
    print("W2 shape= " +  str(parameters["W2"].get_shape()))
'''

'''
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    Xin, X, Y, keep_prob = create_placeholders(28, 28, 1, 10)
    parameters = initialize_parameters()
    Zout = forward_propagation(X, parameters, keep_prob)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Zout,{X: np.random.randn(2,28,28,1), Y: np.random.randn(2,10), keep_prob: 0.8})
    print("Zout= " + str(a))
    print("Zout shape= " + str(a.shape))
'''

'''
    tout = forward_propagation(X, parameters, keep_prob)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(tout,{X: np.random.randn(2,28,28,1), Y: np.random.randn(2,10), keep_prob: 0.8})
    print("Z1 shape= " + str(a[0].shape))
    print("A1 shape= " + str(a[1].shape))
    print("P1 shape= " + str(a[2].shape))
    print("Z2 shape= " + str(a[3].shape))
    print("A2 shape= " + str(a[4].shape))
    print("P2 shape= " + str(a[5].shape))
    print("P2f shape= " + str(a[6].shape))
    print("Z3 shape= " + str(a[7].shape))
    print("Z3d shape= " + str(a[8].shape))
    print("Zout shape= " + str(a[9].shape))
    print("Zout= " + str(a[9]))
    print("Z3[:,0:5]= " + str(a[7]))
    print("Z3d[:,0:5]= " + str(a[8]))
    print("P2[0,0,0,0]= " + str(a[5][0,0,0,0]))
    print("P2f[0,0]= " + str(a[6][0,0]))
    print("P2[1,0,0,0]= " + str(a[5][1,0,0,0]))
    print("P2f[1,0]= " + str(a[6][1,0]))
'''

'''
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    Xin, X, Y, keep_prob = create_placeholders(28, 28, 1, 10)
    parameters = initialize_parameters()
    Zout = forward_propagation(X, parameters, keep_prob)
    cost = compute_cost(Zout, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,28,28,1), Y: np.random.randn(4,10), keep_prob: 0.8})
    print("cost = " + str(a))
'''

#Training
with tf.variable_scope("model"):
    parameters=model(400, batch_size=128, learning_rate=0.001, display_step=10)

#print(parameters)
#print("Global variables:")
#print(tf.global_variables())

'''
print("Calculate test accuracy in function:")
#with tf.variable_scope("model", reuse=True):  use when training is also run
with tf.variable_scope("model"):
    parameters = initialize_parameters()
    test_acc=Test_accuracy(parameters)
    print("test accuracy= " + str(test_acc))
'''














