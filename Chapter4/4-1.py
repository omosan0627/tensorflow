# [MDT-01] 必要なモジュールをインポートして、乱数のシードを設定します。
# In [1]:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20160703)
tf.set_random_seed(20160703)

# [MDT-02] MNISTのデータセットを用意します。
# In [2]:
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

Extracting /tmp/data/train-images-idx3-ubyte.gz
Extracting /tmp/data/train-labels-idx1-ubyte.gz
Extracting /tmp/data/t10k-images-idx3-ubyte.gz
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz

# [MDT-03] 畳込みフィルターが1層のCNNを表現するクラスを定義します。
# In [3]:
class SingleCNN:
    def __init__(self, num_filters, num_units):
        with tf.Graph().as_default():
            self.prepare_model(num_filters, num_units)
            self.prepare_session()

    def prepare_model(self, num_filters, num_units):
        num_units1 = 14*14*num_filters
        num_units2 = num_units
        
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='input')
            x_image = tf.reshape(x, [-1,28,28,1])

        with tf.name_scope('convolution'):
            W_conv = tf.Variable(
                tf.truncated_normal([5,5,1,num_filters], stddev=0.1),
                name='conv-filter')
            h_conv = tf.nn.conv2d(
                x_image, W_conv, strides=[1,1,1,1], padding='SAME',
                name='filter-output')

        with tf.name_scope('pooling'):            
            h_pool =tf.nn.max_pool(h_conv, ksize=[1,2,2,1],
                                   strides=[1,2,2,1], padding='SAME',
                                   name='max-pool')
            h_pool_flat = tf.reshape(h_pool, [-1, 14*14*num_filters],
                                     name='pool-output')

        with tf.name_scope('fully-connected'):
            w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
            b2 = tf.Variable(tf.zeros([num_units2]))
            hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2,
                                 name='fc-output')

        with tf.name_scope('softmax'):
            w0 = tf.Variable(tf.zeros([num_units2, 10]))
            b0 = tf.Variable(tf.zeros([10]))
            p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0,
                              name='softmax-output')
            
        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = -tf.reduce_sum(t * tf.log(p), name='loss')
            train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)
            
        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                              tf.float32), name='accuracy')
            
        tf.scalar_summary("loss", loss)
        tf.scalar_summary("accuracy", accuracy)
        tf.histogram_summary("convolution_filters", W_conv)
        
        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy
        
    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/mnist_df_logs", sess.graph)
        
        self.sess = sess
        self.summary = summary
        self.writer = writer

# [MDT-04] TensorBoard用のデータ出力ディレクトリーを削除して初期化しておきます。
# In [4]:
!rm -rf /tmp/mnist_df_logs

# [MDT-05] パラメーターの最適化を4000回繰り返します。テストセットに対して約98%の正解率が得られます。
# In [5]:
cnn = SingleCNN(16, 1024)

i = 0
for _ in range(4000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    cnn.sess.run(cnn.train_step, feed_dict={cnn.x:batch_xs, cnn.t:batch_ts})
    if i % 50 == 0:
        summary, loss_val, acc_val = cnn.sess.run(
            [cnn.summary, cnn.loss, cnn.accuracy],
            feed_dict={cnn.x:mnist.test.images, cnn.t:mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        cnn.writer.add_summary(summary, i)

# Step: 50, Loss: 3614.847412, Accuracy: 0.891500
# Step: 100, Loss: 2609.521973, Accuracy: 0.922700
# Step: 150, Loss: 1977.392700, Accuracy: 0.943000
# Step: 200, Loss: 1977.111572, Accuracy: 0.941900
# Step: 250, Loss: 1608.331543, Accuracy: 0.953100
# Step: 300, Loss: 1486.694580, Accuracy: 0.956700
# Step: 350, Loss: 1481.067627, Accuracy: 0.957400
# Step: 400, Loss: 1354.234863, Accuracy: 0.958500
# Step: 450, Loss: 1235.755615, Accuracy: 0.961800
# Step: 500, Loss: 1264.820312, Accuracy: 0.960200
# Step: 550, Loss: 1222.289795, Accuracy: 0.960700
# Step: 600, Loss: 1129.764160, Accuracy: 0.964200
# Step: 650, Loss: 922.128540, Accuracy: 0.970400
# Step: 700, Loss: 926.749451, Accuracy: 0.971600
# Step: 750, Loss: 850.130981, Accuracy: 0.973300
# Step: 800, Loss: 1006.377136, Accuracy: 0.968500
# Step: 850, Loss: 902.848633, Accuracy: 0.971600
# Step: 900, Loss: 879.976135, Accuracy: 0.973400
# Step: 950, Loss: 790.658813, Accuracy: 0.974500
# Step: 1000, Loss: 772.311646, Accuracy: 0.976400
# Step: 1050, Loss: 864.686768, Accuracy: 0.973400
# Step: 1100, Loss: 978.713928, Accuracy: 0.970500
# Step: 1150, Loss: 818.460205, Accuracy: 0.974500
# Step: 1200, Loss: 713.533203, Accuracy: 0.978000
# Step: 1250, Loss: 766.665405, Accuracy: 0.977500
# Step: 1300, Loss: 713.059326, Accuracy: 0.977900
# Step: 1350, Loss: 732.855713, Accuracy: 0.978200
# Step: 1400, Loss: 785.117920, Accuracy: 0.976200
# Step: 1450, Loss: 702.009766, Accuracy: 0.978200
# Step: 1500, Loss: 730.830994, Accuracy: 0.977600
# Step: 1550, Loss: 675.383972, Accuracy: 0.979400
# Step: 1600, Loss: 748.971619, Accuracy: 0.976100
# Step: 1650, Loss: 771.830017, Accuracy: 0.976100
# Step: 1700, Loss: 639.565613, Accuracy: 0.980300
# Step: 1750, Loss: 683.713196, Accuracy: 0.979300
# Step: 1800, Loss: 703.339600, Accuracy: 0.979600
# Step: 1850, Loss: 873.175293, Accuracy: 0.975100
# Step: 1900, Loss: 746.795959, Accuracy: 0.976100
# Step: 1950, Loss: 660.269104, Accuracy: 0.981900
# Step: 2000, Loss: 679.535522, Accuracy: 0.978800
# Step: 2050, Loss: 684.502258, Accuracy: 0.980800
# Step: 2100, Loss: 653.159485, Accuracy: 0.980300
# Step: 2150, Loss: 697.510498, Accuracy: 0.978000
# Step: 2200, Loss: 760.059631, Accuracy: 0.976500
# Step: 2250, Loss: 584.984192, Accuracy: 0.982200
# Step: 2300, Loss: 691.510559, Accuracy: 0.978800
# Step: 2350, Loss: 591.455200, Accuracy: 0.981300
# Step: 2400, Loss: 616.852417, Accuracy: 0.980700
# Step: 2450, Loss: 635.980469, Accuracy: 0.981400
# Step: 2500, Loss: 560.255432, Accuracy: 0.983500
# Step: 2550, Loss: 661.358276, Accuracy: 0.980700
# Step: 2600, Loss: 643.725891, Accuracy: 0.980300
# Step: 2650, Loss: 617.790283, Accuracy: 0.981200
# Step: 2700, Loss: 722.376465, Accuracy: 0.978500
# Step: 2750, Loss: 643.536377, Accuracy: 0.981200
# Step: 2800, Loss: 566.617554, Accuracy: 0.982500
# Step: 2850, Loss: 568.770386, Accuracy: 0.982900
# Step: 2900, Loss: 601.478210, Accuracy: 0.982700
# Step: 2950, Loss: 543.404175, Accuracy: 0.983500
# Step: 3000, Loss: 575.665955, Accuracy: 0.983100
# Step: 3050, Loss: 702.199829, Accuracy: 0.979400
# Step: 3100, Loss: 608.132996, Accuracy: 0.982500
# Step: 3150, Loss: 590.421326, Accuracy: 0.982800
# Step: 3200, Loss: 601.763428, Accuracy: 0.982400
# Step: 3250, Loss: 587.208557, Accuracy: 0.983300
# Step: 3300, Loss: 641.174927, Accuracy: 0.981000
# Step: 3350, Loss: 580.049927, Accuracy: 0.982500
# Step: 3400, Loss: 623.968872, Accuracy: 0.981500
# Step: 3450, Loss: 650.404968, Accuracy: 0.982100
# Step: 3500, Loss: 610.692810, Accuracy: 0.982800
# Step: 3550, Loss: 641.231934, Accuracy: 0.982500
# Step: 3600, Loss: 603.037048, Accuracy: 0.982100
# Step: 3650, Loss: 656.173950, Accuracy: 0.980400
# Step: 3700, Loss: 714.270569, Accuracy: 0.980300
# Step: 3750, Loss: 694.605713, Accuracy: 0.982600
# Step: 3800, Loss: 692.162842, Accuracy: 0.981700
# Step: 3850, Loss: 670.902039, Accuracy: 0.981600
# Step: 3900, Loss: 666.686890, Accuracy: 0.981700
# Step: 3950, Loss: 649.617554, Accuracy: 0.981500
# Step: 4000, Loss: 635.322693, Accuracy: 0.983000
