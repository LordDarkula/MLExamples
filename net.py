import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Stores placeholder of unspecified size for training samples of size 784
x = tf.placeholder(tf.float32, [None, 784])

# Creates 10 nodes so outputs are 10 long
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))

# Constructs softmax model
y = tf.nn.softmax(tf.matmul(x, weights) + biases)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
