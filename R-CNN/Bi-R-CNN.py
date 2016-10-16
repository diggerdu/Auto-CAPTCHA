# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.
This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

WORK_DIRECTORY = 'data'
IMAGE_HEIGHT = 20
IMAGE_WIDTH = 80
NUM_CHANNELS = 1
NUM_LABELS = 36
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 1024
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


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])


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
    '''
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    '''
    return conv_net_output_list


cnn_output_list = model(train_data_node, True)
logits_shape = cnn_output_list[0].get_shape().as_list()
print ("cnn_output_shape", logits_shape[0], logits_shape[1])
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

# label_slice = tf.split(split_dim=1, num_split=RNN_STEPS, value=train_labels_node, name='label_slice')

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(final_outputs, train_labels_node))
loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(final_outputs[3], train_labels_node[:,3,:]))
loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(final_outputs[2], train_labels_node[:,2,:]))
loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(final_outputs[1], train_labels_node[:,1,:]))
loss_0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(final_outputs[0], train_labels_node[:,0,:]))

loss = tf.reduce_mean([loss_0, loss_1, loss_2, loss_3])

# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(birnn_out_weights) + tf.nn.l2_loss(birnn_out_biases))

# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0, dtype=data_type())
# Decay once per epoch, using an exponential schedule starting at 0.01.
'''
learning_rate = tf.train.exponential_decay(
  0.01,                # Base learning rate.
  batch * BATCH_SIZE,  # Current index into the dataset.
  ,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)
'''
learning_rate = 0.00003
# Use simple momentum for the optimization.
'''
optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=batch)
'''
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Predictions for the current training minibatch.
#train_prediction = tf.nn.softmax(outputs)

# Predictions for the test and validation, which we'll compute less often.
#eval_prediction = tf.nn.softmax(model(eval_data))


# Small utility function to evaluate a dataset by feeding batches of data to
# {eval_data} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
def eval_in_batches(data, sess):
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_data: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


# Create a local session to run the training.
start_time = time.time()

# loading data
image_data = numpy.load("./image_data.npy")
label_data = numpy.load("./target_data.npy")
print(image_data.shape)
print(label_data.shape)
train_size = 2500
eva_size = image_data.shape[0] - train_size
training_data = image_data[:train_size]
training_label = label_data[:train_size]
eva_data = image_data[train_size:]
eva_label = label_data[train_size:]
with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.

    best = 0;
    for step in xrange(1, NUM_EPOCHS, 1):
        Idx = numpy.random.choice(train_size, BATCH_SIZE)
        batch_data = training_data[Idx]
        batch_labels = training_label[Idx]
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        # Run the graph and fetch some of the nodes.
        output, _, l = sess.run(
            [final_outputs, optimizer, loss],
            feed_dict=feed_dict)
        output = numpy.asarray(output)
        pred = numpy.argmax(output, axis=-1)
        pred = numpy.transpose(pred)
        right_label = numpy.argmax(batch_labels, axis=-1)
        #print (right_label[0][3])
        #print (pred[0][3])
        traning_accurancy = (numpy.sum(numpy.where(numpy.sum(right_label == pred, axis=-1) == 4, 1, 0))) / (BATCH_SIZE * 1.000)
        print ("at epoch {0}, loss is :{1}, accurancy at training set is :{2}".format(step, l, traning_accurancy))

        if step % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.1f ms' %
                  (step, float(step) * BATCH_SIZE / train_size,
                   1000 * elapsed_time / EVAL_FREQUENCY))
            if traning_accurancy > -1:
                Idx = numpy.random.choice(eva_size, EVAL_BATCH_SIZE)
                batch_data = numpy.repeat(eva_data[Idx], 8, axis=0)
                batch_labels = numpy.repeat(eva_label[Idx], 8, axis=0)
                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels}
                output, l = sess.run(
                    [final_outputs, loss],
                    feed_dict=feed_dict)
                output = numpy.asarray(output)
                pred = numpy.argmax(output, axis=-1)
                pred = numpy.transpose(pred)
                right_label = numpy.argmax(batch_labels, axis=-1)
                eva_accurancy = (numpy.sum(numpy.where(numpy.sum(right_label == pred, axis=-1) == 4, 1, 0))) / (
                    BATCH_SIZE * 1.000)
                if eva_accurancy > best:
                    best = eva_accurancy
                print ("at epoch {0}, accurancy in evaluate set is :{1}, loss is :{2}".format(step, eva_accurancy, l))
                print ("############ best accurancy in evaluate set is {0} ###########".format(best))



            # sys.stdout.flush()
    # Finally print the result!
    # test_error = error_rate(eval_in_batches(test_data, sess), test_label)
    # print('Test error: %.1f%%' % test_error)
    # if FLAGS.self_test:
    #    print('test_error', test_error)
    #    assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
    #        test_error,)
