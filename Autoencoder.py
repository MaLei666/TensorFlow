# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# 设置图级随机种子，不同Session中的random系列函数表现出相对协同的特征
tf.set_random_seed(1)

batch_size=64
lr=0.002
n_test_img=10        # 测试图片张数

# mnist数据
mnist=input_data.read_data_sets('MNIST_data',one_hot=False)  #使用非one-hotted数据
test_x=mnist.test.images[:200]
test_y=mnist.test.labels[:200]

tf_x=tf.placeholder(tf.float32,[None,28*28])

# 编码
# dense:全连接层，添加一个层，输入tf_x，输出维数128.激活函数tanh
en0=tf.layers.dense(tf_x,128,tf.nn.tanh)
en1=tf.layers.dense(en0,64,tf.nn.tanh)
en2=tf.layers.dense(en1,12,tf.nn.tanh)
encoded=tf.layers.dense(en2,3)

# 解码
de0=tf.layers.dense(encoded,12,tf.nn.tanh)
de1=tf.layers.dense(de0,64,tf.nn.tanh)
de2=tf.layers.dense(de1,128,tf.nn.tanh)
decoded=tf.layers.dense(de2,28*28,tf.nn.sigmoid)

# labels=真实输出，predictions=预测输出，求误差平方和
loss=tf.losses.mean_squared_error(labels=tf_x,predictions=decoded)
train=tf.train.AdamOptimizer(lr).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

# 创建一个图和一组子图，子图网格的行/列数=2，n_test_img，figsize为整数元组，提供高度和宽度
f,a=plt.subplots(2,n_test_img,figsize=(10,2))
# 打开交互模式
plt.ion()

# 查看初始数据（第一行显示）
view_data=mnist.test.images[:n_test_img]
for i in range(n_test_img):
    a[0][i].imshow(np.reshape(view_data[i],(28,28)),cmap='gray')
    # x、y轴刻度位置列表
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for step in range(8000):
    b_x,b_y=mnist.train.next_batch(batch_size)
    _,encoded_,decoded_,loss_=sess.run([train,encoded,decoded,loss],{tf_x:b_x})

    if step%100==0:
        print('train loss: %.4f' %loss_)
        # 绘制解码后图像（第二行显示）
        decoded_data=sess.run(decoded,{tf_x:view_data})
        for i in range(n_test_img):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoded_data[i],(28,28)),cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.draw()
        plt.pause(0.01)
plt.ioff()

view_data=test_x[:200]
encoded_data=sess.run(encoded,{tf_x:view_data})
# 控制子图默认间距
fig=plt.figure(2)
ax=Axes3D(fig)
X,Y,Z=encoded_data[:,0],encoded_data[:,1],encoded_data[:,2]
for x,y,z,s in zip(X,Y,Z,test_y):
    c=cm.rainbow(int(255*s/9))
    ax.text(x,y,z,s,backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
