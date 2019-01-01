import tensorflow as tf
import numpy as np


# INIT WEIGHT
def init_weight(shape):
    init_random_dist  = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

# INIT BIAIS
def init_biais(shape):
    init_const_biais = tf.constant(0.1,shape=shape)
    return tf.Variable(init_const_biais)

# 2D convolution
def conv2D(x, W):
    #x --> [batch, h, w, c]
    #W --> [filter h, filter w, in channels, out channels]
    return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')

# POOLING
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')

# Convulution layer
def convolution_layer(x, shape):
    W = init_weight(shape)
    b = init_biais([shape[3]])
    return tf.nn.relu(conv2D(x,W) + b)

# fully connected
def fully_layer(x, size):
    input_size = int(x.get_shape()[1])
    W = init_weight([input_size,size])
    b = init_biais([size])
    return tf.matmul(x,W) + b

g = tf.Graph()
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_true = tf.placeholder(tf.float32, shape=[None,10])

    image = tf.reshape(x, [-1,28,28,1])


    conv1 = convolution_layer(image,shape=[5,5, 1, 32])
    conv1_pool = max_pool_2by2(conv1)

    conv2 = convolution_layer(conv1_pool, shape=[5,5,32,64])
    conv2_pool = max_pool_2by2(conv2)
    conv2_flat = tf.reshape(conv2_pool, shape=[-1, 7*7*64])
    fully1 = fully_layer(conv2_flat, size = 1024)


    hold_prob= tf.placeholder(tf.float32)
    fully1_drop = tf.nn.dropout(fully1, keep_prob=hold_prob)

    pred = fully_layer(fully1_drop,10)

    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=pred))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train =  optimizer.minimize(loss_function)

    matches = tf.equal(tf.arg_max(pred,1), tf.arg_max(y_true,1))
    precision = tf.reduce_mean(tf.cast(matches,tf.float32))
    saver = tf.train.Saver()


def predict_cnn(x_im):
    with g.as_default():
        with tf.Session() as sess:
            # Save the variables to disk.
            saver.restore(sess, "./model.ckpt")
            sortie = sess.run(pred, feed_dict={x:x_im, hold_prob:1})
            predit = np.argmax(sortie)
    return predit
