'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict character sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is test code to run on the
8-item data set in the "sample_data" directory, for those without access to TIMIT.

Author: Jon Rein
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np

INPUT_PATH = './training_set/image_data.npy' #directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = './training_set/label_data.npy' #directory of nCharacters 1-D array .npy files

####Load data######
print('Loading data')
image_data = np.load(INPUT_PATH)
label_data = np.load(TARGET_PATH)
totalN = image_data.shape[0]
maxTimeSteps = image_data.shape[1]

####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 30000
batchSize = image_data.shape[0]

####Network Parameters
nFeatures = 22 #12 MFCC coefficients + energy, and derivatives
nHidden = 128
nClasses = 63#62 characters, plus the "blank" for CTC


####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():

    ####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow
        
    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
    inputXrs = tf.reshape(inputX, [-1, nFeatures])
    #  Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    inputList = tf.split(0, maxTimeSteps, inputXrs)
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

    ####Weights & biases
    weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
    weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesOutH2 = tf.Variable(tf.zeros([nHidden]))
    weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                     stddev=np.sqrt(2.0 / nHidden)))
    biasesClasses = tf.Variable(tf.zeros([nClasses]))

    ####Network
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32,
                                       scope='BDLSTM_H1')
    fbH1rs = [tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1]
    outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

    ####Optimizing
    logits3d = tf.pack(logits)
    loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

    ####Evaluating
    logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
                tf.to_float(tf.size(targetY.values))

####Run session
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    im_s = image_data.shape
    batchSeqLengths = np.repeat(im_s[1], im_s[0])
    batchTargetIxs = list()
    for i in range(im_s[0]):
        for j in range(4):
            batchTargetIxs.append([i,j])
    batchTargetIxs = np.array(batchTargetIxs)
    batchTargetVals = label_data.reshape(label_data.shape[0] * label_data.shape[1])
    batchTargetShape = np.array([im_s[0], im_s[1]])
    batchInputs = image_data.reshape((im_s[1], im_s[0], im_s[2]))
    print(batchTargetIxs.shape, batchTargetVals.shape, batchTargetShape)
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        print(batchTargetVals.shape)
        print(batchTargetIxs.shape)
        feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
        _, l, er, lmt, output_sparse_tensor = session.run([optimizer, loss, errorRate, logitsMaxTest, predictions], feed_dict=feedDict)
        print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
        print('loss:', l)
        print(output_sparse_tensor.values)
        print(output_sparse_tensor.values.shape)
        print(output_sparse_tensor.shape)
        print('error rate:', er)
        batchErrors = er*len(batchSeqLengths)
        epochErrorRate = batchErrors / totalN
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)