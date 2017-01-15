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

# Creates placeholder for true labels
y_ = tf.placeholder(tf.float32, [None, 10])

# Takes average of -sum(y_actual * log(y_predicted)) for all predictions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
