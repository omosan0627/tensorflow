# [MDS-01] モジュールをインポートします。
# In [1]:
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# [MDS-02] MNISTのデータセットをダウンロードして、オブジェクトに格納します。
# In [2]:
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
# Extracting /tmp/data/train-images-idx3-ubyte.gz
# Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
# Extracting /tmp/data/train-labels-idx1-ubyte.gz
# Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
# Extracting /tmp/data/t10k-images-idx3-ubyte.gz
# Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
# Extracting /tmp/data/t10k-labels-idx1-ubyte.gz

# [MDS-03] トレーニングセットから、10個分のデータを取り出して、画像データとラベルを別々の変数に格納します。
# In [3]:
images, labels = mnist.train.next_batch(10)

# [MDS-04] 1つめの画像データを確認します。各ピクセルの濃度が並んだリスト（arrayオブジェクト）になっています。
# In [4]:
print images[0]

# [ 0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.38039219  0.37647063
#   0.3019608   0.46274513  0.2392157   0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.35294119  0.5411765
#   0.92156869  0.92156869  0.92156869  0.92156869  0.92156869  0.92156869
#   0.98431379  0.98431379  0.97254908  0.99607849  0.96078438  0.92156869
#   0.74509805  0.08235294  0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.
#   0.54901963  0.98431379  0.99607849  0.99607849  0.99607849  0.99607849
#   0.99607849  0.99607849  0.99607849  0.99607849  0.99607849  0.99607849
#   0.99607849  0.99607849  0.99607849  0.99607849  0.74117649  0.09019608
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.88627458  0.99607849  0.81568635
#   0.78039223  0.78039223  0.78039223  0.78039223  0.54509807  0.2392157
#   0.2392157   0.2392157   0.2392157   0.2392157   0.50196081  0.8705883
#   0.99607849  0.99607849  0.74117649  0.08235294  0.          0.          0.
#   0.          0.          0.          0.          0.          0.
#   0.14901961  0.32156864  0.0509804   0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.13333334  0.83529419  0.99607849  0.99607849  0.45098042  0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.32941177  0.99607849  0.99607849  0.91764712  0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.32941177  0.99607849  0.99607849  0.91764712  0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.41568631  0.6156863   0.99607849  0.99607849  0.95294124  0.20000002
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.09803922  0.45882356  0.89411771
#   0.89411771  0.89411771  0.99215692  0.99607849  0.99607849  0.99607849
#   0.99607849  0.94117653  0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.26666668  0.4666667   0.86274517
#   0.99607849  0.99607849  0.99607849  0.99607849  0.99607849  0.99607849
#   0.99607849  0.99607849  0.99607849  0.55686277  0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.14509805  0.73333335  0.99215692
#   0.99607849  0.99607849  0.99607849  0.87450987  0.80784321  0.80784321
#   0.29411766  0.26666668  0.84313732  0.99607849  0.99607849  0.45882356
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.44313729
#   0.8588236   0.99607849  0.94901967  0.89019614  0.45098042  0.34901962
#   0.12156864  0.          0.          0.          0.          0.7843138
#   0.99607849  0.9450981   0.16078432  0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.66274512  0.99607849  0.6901961   0.24313727  0.          0.
#   0.          0.          0.          0.          0.          0.18823531
#   0.90588242  0.99607849  0.91764712  0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.07058824  0.48627454  0.          0.          0.
#   0.          0.          0.          0.          0.          0.
#   0.32941177  0.99607849  0.99607849  0.65098041  0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.54509807  0.99607849  0.9333334   0.22352943  0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.
#   0.82352948  0.98039222  0.99607849  0.65882355  0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.94901967  0.99607849  0.93725497  0.22352943  0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.
#   0.34901962  0.98431379  0.9450981   0.33725491  0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.
#   0.01960784  0.80784321  0.96470594  0.6156863   0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.01568628  0.45882356  0.27058825  0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.
#   0.          0.          0.          0.          0.          0.          0.        ]


# [MDS-05] 対応するラベルを確認します。先頭を0として、7番目の要素が1になっているので、「7」の画像である事を示します。
# In [5]:
print labels[0]

[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]

# [MDS-06] 画像データを実際の画像として表示してみます。
# In [6]:
fig = plt.figure(figsize=(8,4))
for c, (image, label) in enumerate(zip(images, labels)):
    subplot = fig.add_subplot(2,5,c+1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(label))
    subplot.imshow(image.reshape((28,28)), vmin=0, vmax=1,
                   cmap=plt.cm.gray_r, interpolation="nearest")

 
