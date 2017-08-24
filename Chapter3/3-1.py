# [DNE-01] モジュールをインポートして、乱数のシードを設定します。
# In [1]:
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal, permutation
import pandas as pd
from pandas import DataFrame, Series

np.random.seed(20160615)
tf.set_random_seed(20160615)

# [DNE-02] トレーニングセットのデータを生成します。
# In [2]:
def generate_datablock(n, mu, var, t):
    data = multivariate_normal(mu, np.eye(2)*var, n)
    df = DataFrame(data, columns=['x1','x2'])
    df['t'] = t
    return df

df0 = generate_datablock(30, [-7,-7], 18, 1)
df1 = generate_datablock(30, [-7,7], 18, 0)
df2 = generate_datablock(30, [7,-7], 18, 0)
df3 = generate_datablock(30, [7,7], 18, 1)

df = pd.concat([df0, df1, df2, df3], ignore_index=True)
train_set = df.reindex(permutation(df.index)).reset_index(drop=True)

# [DNE-03] (x1, x2) と t を別々に集めたものをNumPyのarrayオブジェクトとして取り出しておきます。
# In [3]:
train_x = train_set[['x1','x2']].as_matrix()
train_t = train_set['t'].as_matrix().reshape([len(train_set), 1])

# [DNE-04] 二層ネットワークによる二項分類器のモデルを定義します。
# In [4]:
num_units1 = 2
num_units2 = 2

x = tf.placeholder(tf.float32, [None, 2])

w1 = tf.Variable(tf.truncated_normal([2, num_units1]))
b1 = tf.Variable(tf.zeros([num_units1]))
hidden1 = tf.nn.tanh(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.zeros([num_units2]))
hidden2 = tf.nn.tanh(tf.matmul(hidden1, w2) + b2)

w0 = tf.Variable(tf.zeros([num_units2, 1]))
b0 = tf.Variable(tf.zeros([1]))
p = tf.nn.sigmoid(tf.matmul(hidden2, w0) + b0)

# [DNE-05] 誤差関数 loss、トレーニングアルゴリズム train_step、正解率 accuracy を定義します。
# In [5]:
t = tf.placeholder(tf.float32, [None, 1])
loss = -tf.reduce_sum(t*tf.log(p) + (1-t)*tf.log(1-p))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# [DNE-06] セッションを用意して、Variableを初期化します。
# In [6]:
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# [DNE-07] パラメーターの最適化を2000回繰り返します。
# In [7]:
i = 0
for _ in range(2000):
    i += 1
    sess.run(train_step, feed_dict={x:train_x, t:train_t})
    if i % 100 == 0:
        loss_val, acc_val = sess.run(
            [loss, accuracy], feed_dict={x:train_x, t:train_t})
        print ('Step: %d, Loss: %f, Accuracy: %f'
               % (i, loss_val, acc_val))

# Step: 100, Loss: 83.176933, Accuracy: 0.508333
# Step: 200, Loss: 83.176178, Accuracy: 0.508333
# Step: 300, Loss: 83.174591, Accuracy: 0.508333
# Step: 400, Loss: 83.171082, Accuracy: 0.500000
# Step: 500, Loss: 83.162636, Accuracy: 0.508333
# Step: 600, Loss: 83.140877, Accuracy: 0.516667
# Step: 700, Loss: 83.075996, Accuracy: 0.541667
# Step: 800, Loss: 82.822495, Accuracy: 0.541667
# Step: 900, Loss: 81.475693, Accuracy: 0.625000
# Step: 1000, Loss: 75.140419, Accuracy: 0.658333
# Step: 1100, Loss: 59.051060, Accuracy: 0.866667
# Step: 1200, Loss: 46.646378, Accuracy: 0.900000
# Step: 1300, Loss: 41.770844, Accuracy: 0.900000
# Step: 1400, Loss: 39.639244, Accuracy: 0.900000
# Step: 1500, Loss: 38.510742, Accuracy: 0.900000
# Step: 1600, Loss: 37.788445, Accuracy: 0.900000
# Step: 1700, Loss: 37.159111, Accuracy: 0.900000
# Step: 1800, Loss: 36.648502, Accuracy: 0.900000
# Step: 1900, Loss: 36.529396, Accuracy: 0.891667
# Step: 2000, Loss: 36.352604, Accuracy: 0.891667

# [DNE-08] 得られた確率を色の濃淡で図示します。
# In [8]:
train_set1 = train_set[train_set['t']==1]
train_set2 = train_set[train_set['t']==0]

fig = plt.figure(figsize=(12,12))
subplot = fig.add_subplot(1,1,1)
subplot.set_ylim([-15,15])
subplot.set_xlim([-15,15])
subplot.scatter(train_set1.x1, train_set1.x2, marker='x')
subplot.scatter(train_set2.x1, train_set2.x2, marker='o')

locations = []
for x2 in np.linspace(-15,15,100):
    for x1 in np.linspace(-15,15,100):
        locations.append((x1,x2))
p_vals = sess.run(p, feed_dict={x:locations})
p_vals = p_vals.reshape((100,100))
subplot.imshow(p_vals, origin='lower', extent=(-15,15,-15,15),
               cmap=plt.cm.gray_r, alpha=0.5)

plt.show()
# Out[8]:
# <matplotlib.image.AxesImage at 0x5aefe50>
