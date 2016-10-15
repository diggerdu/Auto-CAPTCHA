import tensorflow as tf
import numpy as np

ph = tf.placeholder(shape=[None,3,3], dtype=tf.int32)

#x = tf.slice(ph, [0, 0], [3, 2])
x = ph[:,0:1,:]
input_ = np.reshape(np.arange(27),(3,3,3))
print input_.shape
print input_

with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print sess.run(x, feed_dict={ph: input_})
