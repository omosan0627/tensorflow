# [MDC-01] 必要なモジュールをインポートして、乱数のシードを設定します。
# In [1]:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20160703)
tf.set_random_seed(20160703)

# [MDC-02] MNISTのデータセットを用意します。
# In [2]:
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Extracting /tmp/data/train-images-idx3-ubyte.gz
# Extracting /tmp/data/train-labels-idx1-ubyte.gz
# Extracting /tmp/data/t10k-images-idx3-ubyte.gz
# Extracting /tmp/data/t10k-labels-idx1-ubyte.gz

# [MDC-03] フィルターに対応する Variable を用意して、入力データにフィルターとプーリング層を適用する計算式を定義します。
# In [3]:
num_filters = 16

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

W_conv = tf.Variable(tf.truncated_normal([5,5,1,num_filters],
                                         stddev=0.1))
h_conv = tf.nn.conv2d(x_image, W_conv,
                      strides=[1,1,1,1], padding='SAME')
h_pool =tf.nn.max_pool(h_conv, ksize=[1,2,2,1],
                       strides=[1,2,2,1], padding='SAME')

# [MDC-04] プーリング層からの出力を全結合層を経由してソフトマックス関数に入力する計算式を定義します。
# In [4]:
h_pool_flat = tf.reshape(h_pool, [-1, 14*14*num_filters])

num_units1 = 14*14*num_filters
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool_flat, w2) + b2)

w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden2, w0) + b0)

# [MDC-05] 誤差関数 loss、トレーニングアルゴリズム train_step、正解率 accuracy を定義します。
# In [5]:
t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer(0.0005).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# [MDC-06] セッションを用意して、Variable を初期化します。
# In [6]:
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()

# [MDC-07] パラメーターの最適化を4000回繰り返します。
# 最終的に、テストセットに対して約98%の正解率が得られます。
# In [7]:
i = 0
for _ in range(4000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
            feed_dict={x:mnist.test.images, t:mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))
        saver.save(sess, 'mdc_session', global_step=i)

# Step: 100, Loss: 2726.630615, Accuracy: 0.917900
# Step: 200, Loss: 2016.798096, Accuracy: 0.943700
# Step: 300, Loss: 1600.125977, Accuracy: 0.953200
# Step: 400, Loss: 1449.618408, Accuracy: 0.955600
# Step: 500, Loss: 1362.578125, Accuracy: 0.956200
# Step: 600, Loss: 1135.334595, Accuracy: 0.965200
# Step: 700, Loss: 999.617493, Accuracy: 0.969300
# Step: 800, Loss: 972.449707, Accuracy: 0.970200
# Step: 900, Loss: 941.483398, Accuracy: 0.968800
# Step: 1000, Loss: 886.782104, Accuracy: 0.973500
# Step: 1100, Loss: 921.191101, Accuracy: 0.973200
# Step: 1200, Loss: 691.343445, Accuracy: 0.978000
# Step: 1300, Loss: 727.946289, Accuracy: 0.977300
# Step: 1400, Loss: 761.936218, Accuracy: 0.976200
# Step: 1500, Loss: 742.681763, Accuracy: 0.978200
# Step: 1600, Loss: 924.516724, Accuracy: 0.971500
# Step: 1700, Loss: 670.436218, Accuracy: 0.980800
# Step: 1800, Loss: 655.680481, Accuracy: 0.980500
# Step: 1900, Loss: 792.281738, Accuracy: 0.975600
# Step: 2000, Loss: 723.777954, Accuracy: 0.978200
# Step: 2100, Loss: 635.388245, Accuracy: 0.980800
# Step: 2200, Loss: 747.225708, Accuracy: 0.977300
# Step: 2300, Loss: 701.824646, Accuracy: 0.980000
# Step: 2400, Loss: 669.409058, Accuracy: 0.979800
# Step: 2500, Loss: 701.669739, Accuracy: 0.980200
# Step: 2600, Loss: 793.589294, Accuracy: 0.976700
# Step: 2700, Loss: 677.845093, Accuracy: 0.980900
# Step: 2800, Loss: 654.192322, Accuracy: 0.981800
# Step: 2900, Loss: 667.308777, Accuracy: 0.980400
# Step: 3000, Loss: 765.483276, Accuracy: 0.976600
# Step: 3100, Loss: 646.766357, Accuracy: 0.981300
# Step: 3200, Loss: 693.853271, Accuracy: 0.980300
# Step: 3300, Loss: 738.400208, Accuracy: 0.980700
# Step: 3400, Loss: 708.065308, Accuracy: 0.980700
# Step: 3500, Loss: 701.663330, Accuracy: 0.980300
# Step: 3600, Loss: 656.354309, Accuracy: 0.981400
# Step: 3700, Loss: 671.281555, Accuracy: 0.981300
# Step: 3800, Loss: 731.150269, Accuracy: 0.981000
# Step: 3900, Loss: 708.207214, Accuracy: 0.982400
# Step: 4000, Loss: 708.660156, Accuracy: 0.980400

# [MDC-08] セッション情報を保存したファイルが生成されていることを確認します。
# In [8]:
# !ls mdc_session*

# mdc_session-3600       mdc_session-3800       mdc_session-4000
# mdc_session-3600.meta  mdc_session-3800.meta  mdc_session-4000.meta
# mdc_session-3700       mdc_session-3900
# mdc_session-3700.meta  mdc_session-3900.meta

