from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time
import argparse
import string

import numpy
from PIL import Image
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


parser = argparse.ArgumentParser(description='Auto-CAPTCHA')
parser.add_argument('file_name', type=str, help='the file name of validate code image')
file_name = parser.parse_args().file_name

decode_table = dict(enumerate(list(string.lowercase) + [str(i) for i in range(10)]))


WORK_DIRECTORY = 'data'
IMAGE_HEIGHT = 20
IMAGE_WIDTH = 80
NUM_CHANNELS = 1
NUM_LABELS = 36
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 1
NUM_EPOCHS = 100000000000
EVAL_BATCH_SIZE = 128
EVAL_FREQUENCY = 4  # Number of steps between evaluations.
RNN_STEPS = 4
RNN_UNIT_NUM = 512
RNN_DEPTH = 3

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
tf.app.flags.DEFINE_boolean('use_fp16', False, "Use half floats instead of full floats if True.")
FLAGS = tf.app.flags.FLAGS


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32


# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
train_data_node = tf.placeholder(data_type(), shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT))
train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE, RNN_STEPS, NUM_LABELS))
eval_data = tf.placeholder(data_type(), shape=(EVAL_BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT))

# The variables below hold all the trainable weights. They are passed an
# initial value which will be assigned when we call:
# {tf.initialize_all_variables().run()}
conv1_weights = tf.Variable(
    tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                        stddev=0.1,
                        seed=SEED, dtype=data_type()))
conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
conv2_weights = tf.Variable(tf.truncated_normal(
    [5, 5, 32, 64], stddev=0.1,
    seed=SEED, dtype=data_type()))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
fc1_weights = tf.Variable(tf.truncated_normal([1600, 512], stddev=0.1,seed=SEED,dtype=data_type()))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
birnn_out_weights = tf.Variable(tf.truncated_normal([RNN_UNIT_NUM * 2, NUM_LABELS], stddev=0.1, seed=SEED, dtype=data_type()))
birnn_out_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))


# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, train=False):
    #data = tf.nn.dropout(data, keep_prob=0.5)
    data_shape = data.get_shape().as_list()
    data = tf.reshape(data, (data_shape[0], data_shape[1], data_shape[2], NUM_CHANNELS))
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv_1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
    # Bias and rectified linear non-linearity.
    relu_1 = tf.nn.relu(tf.nn.bias_add(conv_1, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool_1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv_2 = tf.nn.conv2d(pool_1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
    relu_2 = tf.nn.relu(tf.nn.bias_add(conv_2, conv2_biases))
    pool_2 = tf.nn.max_pool(relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool_2.get_shape().as_list()
    pool_output_list = tf.split(split_dim=1, num_split=RNN_STEPS, value=pool_2, name='pool_2_output')
    conv_net_output_list = list()
    for pool_output in pool_output_list:
        pool_output = tf.reshape(pool_output, [pool_shape[0], int(pool_shape[1] * pool_shape[2] * pool_shape[3] / RNN_STEPS)])
        conv_net_output_list.append(tf.nn.relu(tf.matmul(pool_output, fc1_weights) + fc1_biases))
    assert len(conv_net_output_list) == RNN_STEPS
    return conv_net_output_list


cnn_output_list = model(train_data_node, True)
rnn_layer = tf.nn.rnn_cell.BasicRNNCell(RNN_UNIT_NUM)
rnn_fw_net = rnn_cell.MultiRNNCell([rnn_layer] * RNN_DEPTH)
rnn_bw_net = rnn_cell.MultiRNNCell([rnn_layer] * RNN_DEPTH)

# rnn_outputs, states = rnn.rnn(rnn_net, cnn_output_list, dtype=tf.float32)
rnn_outputs = rnn.bidirectional_rnn(rnn_fw_net, rnn_bw_net, cnn_output_list, dtype=tf.float32)

final_outputs = list()
for item in rnn_outputs[0]:
    rnn_output = tf.sigmoid(tf.matmul(item, birnn_out_weights) + birnn_out_biases)
    #rnn_output = tf.nn.dropout(rnn_output, 0.5, seed=SEED)
    final_outputs.append(rnn_output)


saver = tf.train.Saver()

# Create a local session to run the training.
with tf.Session() as sess:
    image_data = list()
    im = numpy.asarray(Image.open(file_name).convert('L'))
    im = (im -120) / (255.000 - 120)
    for start in range(3, 40, 12):
        tmp_single = im[1:21, start:start+20]
        image_data.append(tmp_single)
    image_data = numpy.asarray(image_data)
    image_data = numpy.reshape(image_data, (1, 80, 20))
    
    label_data = numpy.ones((1, 4, 36))

    saver.restore(sess, "checkpoint/model.ckpt")
    sparse_mat = sess.run(final_outputs, feed_dict = {train_data_node: image_data, train_labels_node: label_data})
    pred = numpy.argmax(sparse_mat, axis=-1)
    print (pred)
    print (map(lambda code: decode_table[code], pred.reshape(4,).tolist()))





