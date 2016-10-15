'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf


'''
loading data
'''
image_data = np.load("training_set/image_data.npy")
target_data = np.load("training_set/target_data.npy")


# Parameters
learning_rate = 0.001
training_epochs = 10000000
batch_size = 4096
display_step = 50

# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features
n_input = 440 # MNIST data input (img shape: 28*28)
n_classes = 62 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# Define loss and optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()




validate_x = image_data[9000:9364]
validate_x = validate_x.reshape(validate_x.shape[0], 440)
print (validate_x.shape)
validate_label = target_data[9000:9364]
validate_y = np.zeros((364, 62))
for i in range(364):
    validate_y[i][validate_label[i]] = 1

np.save("fuckimg", validate_x[18])
np.save("fucklabel", validate_y[18])
best = 0.90
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        idx = np.random.choice(9200, batch_size)
        batch_x = image_data[idx]
        batch_x = batch_x.reshape(batch_size, 440)
        batch_label = target_data[idx]
        batch_y = np.zeros((batch_size, 62))
        for i in range(batch_size):
            batch_y[i][batch_label[i]] = 1

        if epoch % display_step == 0:
            acc = accuracy.eval({x:validate_x, y: validate_y})
            train_acc = accuracy.eval({x:batch_x, y: batch_y})
            if train_acc > 0.99:
                best = acc
                saver.save(sess, 'model.ckpt')
            print("Thus far Accuracy:", acc, " Best:", best)
            print("Traning accuracy:", train_acc)
            
        else:
            # Run optimization op (backprop) and cost op (to get loss value)
            _ = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
