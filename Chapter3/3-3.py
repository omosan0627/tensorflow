# [MSL-01] 必要なモジュールをインポートして、乱数のシードを設定します。
# In [1]:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20160612)
tf.set_random_seed(20160612)

# [MSL-02] MNISTのデータセットを用意します。
# In [2]:
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

Extracting /tmp/data/train-images-idx3-ubyte.gz
Extracting /tmp/data/train-labels-idx1-ubyte.gz
Extracting /tmp/data/t10k-images-idx3-ubyte.gz
Extracting /tmp/data/t10k-labels-idx1-ubyte.gz

# [MSL-03] 単層ニューラルネットワークを用いた確率 p の計算式を用意します。
# In [3]:
num_units = 1024

x = tf.placeholder(tf.float32, [None, 784])

w1 = tf.Variable(tf.truncated_normal([784, num_units]))
b1 = tf.Variable(tf.zeros([num_units]))
hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w0 = tf.Variable(tf.zeros([num_units, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)

# [MSL-04] 誤差関数 loss、トレーニングアルゴリズム train_step、正解率 accuracy を定義します。
# In [4]:
t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer().minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# [MSL-05] セッションを用意して、Variableを初期化します。
# In [5]:
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# [MSL-06] パラメーターの最適化を2000回繰り返します。
# 1回の処理において、トレーニングセットから取り出した100個のデータを用いて、勾配降下法を適用します。
# 最終的に、テストセットに対して約97%の正解率が得られます。
# In [6]:
i = 0
for _ in range(2000):
    i += 1
    batch_xs, batch_ts = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    if i % 100 == 0:
        loss_val, acc_val = sess.run([loss, accuracy],
            feed_dict={x:mnist.test.images, t: mnist.test.labels})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))

# Step: 100, Loss: 3136.286377, Accuracy: 0.906700
# Step: 200, Loss: 2440.697021, Accuracy: 0.928000
# Step: 300, Loss: 1919.005249, Accuracy: 0.941900
# Step: 400, Loss: 1982.860718, Accuracy: 0.939400
# Step: 500, Loss: 1734.469971, Accuracy: 0.945500
# Step: 600, Loss: 1377.535767, Accuracy: 0.956100
# Step: 700, Loss: 1332.846313, Accuracy: 0.960600
# Step: 800, Loss: 1184.055786, Accuracy: 0.963600
# Step: 900, Loss: 1134.486084, Accuracy: 0.964700
# Step: 1000, Loss: 1236.647095, Accuracy: 0.961900
# Step: 1100, Loss: 1116.422852, Accuracy: 0.965500
# Step: 1200, Loss: 1125.365234, Accuracy: 0.964700
# Step: 1300, Loss: 1193.366577, Accuracy: 0.961900
# Step: 1400, Loss: 1101.243652, Accuracy: 0.966800
# Step: 1500, Loss: 1062.339966, Accuracy: 0.969400
# Step: 1600, Loss: 1112.656494, Accuracy: 0.966600
# Step: 1700, Loss: 953.149780, Accuracy: 0.972200
# Step: 1800, Loss: 960.959900, Accuracy: 0.970900
# Step: 1900, Loss: 1035.524414, Accuracy: 0.967900
# Step: 2000, Loss: 990.451965, Accuracy: 0.970600

# [MSL-07] 最適化されたパラメーターを用いて、テストセットに対する予測を表示します。
# ここでは、「０」〜「９」の数字に対して、正解と不正解の例を３個ずつ表示します。
# In [7]:
images, labels = mnist.test.images, mnist.test.labels
p_val = sess.run(p, feed_dict={x:images, t: labels}) 

fig = plt.figure(figsize=(8,15))
for i in range(10):
    c = 1
    for (image, label, pred) in zip(images, labels, p_val):
        prediction, actual = np.argmax(pred), np.argmax(label)
        if prediction != i:
            continue
        if (c < 4 and i == actual) or (c >= 4 and i != actual):
            subplot = fig.add_subplot(10,6,i*6+c)
            subplot.set_xticks([])
            subplot.set_yticks([])
            subplot.set_title('%d / %d' % (prediction, actual))
            subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1,
                           cmap=plt.cm.gray_r, interpolation="nearest")
            c += 1
            if c > 6:
                break
