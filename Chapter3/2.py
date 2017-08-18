# [MST-01] モジュールをインポートして、乱数のシードを設定します。
# In [1]:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20160612)
tf.set_random_seed(20160612)

# [MST-02] MNISTのデータセットを用意します。
# In [2]:
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

Extracting /tmp/data/train-images-idx3-ubyte.gz
Extracting /tmp/data/train-labels-idx1-ubyte.gz
Extracting /tmp/data/t10k-images-idx3-ubyte.gz
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz

# [MST-03] 単層ニューラルネットワークを表現するクラスを定義します。
# In [3]:
class SingleLayerNetwork:
    def __init__(self, num_units):
        with tf.Graph().as_default():
            self.prepare_model(num_units)
            self.prepare_session()

    def prepare_model(self, num_units):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='input')
        
        with tf.name_scope('hidden'):
            w1 = tf.Variable(tf.truncated_normal([784, num_units]),
                             name='weights')        
            b1 = tf.Variable(tf.zeros([num_units]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='hidden1')
        
        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units, 10]), name='weights')
            b0 = tf.Variable(tf.zeros([10]), name='biases')
            p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0, name='softmax')

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = -tf.reduce_sum(t * tf.log(p), name='loss')
            train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                              tf.float32), name='accuracy')

        tf.scalar_summary("loss", loss)
        tf.scalar_summary("accuracy", accuracy)
        tf.histogram_summary("weights_hidden", w1)
        tf.histogram_summary("biases_hidden", b1)
        tf.histogram_summary("weights_output", w0)
        tf.histogram_summary("biases_output", b0)
                
        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy
            
    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/mnist_sl_logs", sess.graph)
        
        self.sess = sess
        self.summary = summary
        self.writer = writer

# [MST-04] TensorBoard用のデータ出力ディレクトリーを削除して初期化しておきます。
# In [4]:
!rm -rf /tmp/mnist_sl_logs

# [MST-05] パラメーターの最適化を2000回繰り返します。テストセットに対して、約97%の正解率が得られます。
# In [5]:
nn = SingleLayerNetwork(1024)

i = 0
for _ in range(2000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts})
    if i % 100 == 0:
        summary, loss_val, acc_val = nn.sess.run(
            [nn.summary, nn.loss, nn.accuracy],
            feed_dict={nn.x:mnist.test.images, nn.t: mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        nn.writer.add_summary(summary, i)

# Step: 100, Loss: 3184.660156, Accuracy: 0.901100
# Step: 200, Loss: 2317.262207, Accuracy: 0.931200
# Step: 300, Loss: 1844.382080, Accuracy: 0.944200
# Step: 400, Loss: 1852.014038, Accuracy: 0.942600
# Step: 500, Loss: 1652.209106, Accuracy: 0.946800
# Step: 600, Loss: 1368.317261, Accuracy: 0.956900
# Step: 700, Loss: 1287.111450, Accuracy: 0.960100
# Step: 800, Loss: 1182.710205, Accuracy: 0.962600
# Step: 900, Loss: 1126.482056, Accuracy: 0.965800
# Step: 1000, Loss: 1357.330322, Accuracy: 0.959800
# Step: 1100, Loss: 1068.415649, Accuracy: 0.967500
# Step: 1200, Loss: 1094.205078, Accuracy: 0.967600
# Step: 1300, Loss: 1215.297729, Accuracy: 0.961900
# Step: 1400, Loss: 1033.312012, Accuracy: 0.970100
# Step: 1500, Loss: 1163.462158, Accuracy: 0.964600
# Step: 1600, Loss: 1092.042358, Accuracy: 0.965700
# Step: 1700, Loss: 939.083984, Accuracy: 0.970300
# Step: 1800, Loss: 971.933289, Accuracy: 0.969600
# Step: 1900, Loss: 1001.808167, Accuracy: 0.970000
# Step: 2000, Loss: 971.080200, Accuracy: 0.970800
