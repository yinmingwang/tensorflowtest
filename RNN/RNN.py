from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
SUMMARY_DIR = "/log/RNN"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
lr = 0.001
input_size = 28
timestep_size = 28
hidden_size = 28
layer_num = 2
class_num = 10
X_ = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32,[])
batch_size = tf.placeholder(tf.int32,[])
X = tf.reshape(X_,[-1,28,28])

def lstm_cell():
    cell = rnn.LSTMCell(hidden_size,reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)

mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple = True)
init_state = mlstm_cell.zero_state(batch_size,dtype=tf.float32)
#output,state = tf.nn.dynamic_rnn(mlstm_cell,input= X,initial_state=init_state,time_major=False)
#h_state = state[-1][1]
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        (cell_output,state) = mlstm_cell(X[:,timestep,:],state)
        outputs.append(cell_output)
h_state = outputs[-1]

W = tf.Variable(tf.truncated_normal([hidden_size,class_num],stddev=0.1),dtype=tf.float32)
tf.summary.histogram('Weight',W)
b = tf.Variable(tf.constant(0.1,shape=[class_num]),dtype=tf.float32)
tf.summary.histogram('Bias',b)
y_pre = tf.nn.softmax(tf.matmul(h_state,W)+b)

cross_entropy = -tf.reduce_mean(y*tf.log(y_pre))
tf.summary.scalar('cross_entropy',cross_entropy)
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
tf.summary.scalar('accuracy',accuracy)

with tf.Session() as sess:
    merged_all = tf.summary.merge_all()
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(2000):
        x_train,y_train = mnist.train.next_batch(128)
        if i%100 == 0:
           train_accuracy = sess.run(accuracy,feed_dict={X_:x_train,y:y_train,keep_prob:1.0,batch_size:128})
           print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, i, train_accuracy))
        _, summary = sess.run([train_op,merged_all],feed_dict={X_:x_train,y:y_train,keep_prob:0.8,batch_size:128})
        summary_writer.add_summary(summary,i)
    saver.save(sess, './save/model.ckpt')
    summary_writer.close()
    print('test accuracy %g' % sess.run(accuracy,feed_dict={X_:mnist.test.images,y:mnist.test.labels,keep_prob:1.0,batch_size:mnist.test.images.shape[0]}))



