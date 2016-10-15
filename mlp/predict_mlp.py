'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
from PIL import Image 
import numpy as np
import tensorflow as tf
import string
import sys


'''
loading data
'''
char_table = dict()
idx = 0;

for char in string.uppercase:
    char_table[char] = idx
    idx += 1

for char in string.lowercase:
    char_table[char] = idx
    idx += 1

for i in range(10):
    char_table[str(i)] = idx
    idx += 1

image_data = list()
im = np.asarray(Image.open(sys.argv[1]).convert('L'))
im = im - 120
im = im / (255.000 - 120)
for start in range(3, 40, 12):
	tmp = im[:,start:start+20]
	tmp = np.reshape(tmp, (1,440))
	image_data.append(tmp)
#target_data = np.load("training_set/target_data.npy")


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




# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "./model.ckpt")
    for entry in image_data:
	label = np.argmax(pred.eval({x: entry}))
        print (char_table.keys()[char_table.values().index(label)])
