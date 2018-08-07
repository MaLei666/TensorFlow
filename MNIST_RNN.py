8# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# 参数定义
lr=0.01                # 学习率
batch_size=64
inputs_size=28            # mnist数据 宽度（28*28）
time_steps=28             # 数据 高度

# 导入MNIST手写数字库
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
# 数据
test_x=mnist.test.images[:2000]
test_y=mnist.test.labels[:2000]

# matplotlib显示一个图片
# print(mnist.train.images.shape) #(55000,28*28)
# print(mnist.train.labels.shape) #(55000,10)
# plt.imshow(mnist.train.images[10].reshape((28,28)),cmap='gray')
# plt.title('%i'%np.argmax(mnist.train.labels[10]))
# plt.show()

# 设置TensorFlow占位符
tf_x=tf.placeholder(tf.float32,[None,time_steps*inputs_size]) #shape（batch，784）
image=tf.reshape(tf_x,[-1,time_steps,inputs_size])  #（batch,height,width,channel）
tf_y=tf.placeholder(tf.int32,[None,10])
# 定义rnn主体结构
# cell计算
# 对于 lstm 来说, state可被分为(c_state, h_state)
cell=tf.contrib.rnn.BasicLSTMCell(num_units=64)
# 初始化全零state
init_state=cell.zero_state(batch_size,dtype=tf.float32)

# 使用tf.nn.dynamic_rnn(cell, inputs), 要确定 inputs 的格式.
# 函数中的 time_major 参数会针对不同 inputs 格式有不同的值.
outputs,(h_c,h_n)=tf.nn.dynamic_rnn(
    cell,            #选择的cell
    image,           # 输入
    initial_state=None,  #初始隐含层
    dtype=tf.float32,   # 如果隐含层设置为none，必须给dtype一个值
    time_major=False # False: (batch, time step, input); True: (time step, batch, input)
)

output=tf.layers.dense(outputs[:,-1,:],10)

loss=tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)
train_op=tf.train.AdamOptimizer(lr).minimize(loss)
accuracy=tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(output,axis=1),)[1]

sess=tf.Session()
init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(init_op)
for step in range(1200):
    b_X,b_y=mnist.train.next_batch(batch_size)
    _,loss_=sess.run([train_op,loss],{tf_x:b_X,tf_y:b_y})
    if step%50 ==0:
        accuracy_=sess.run(accuracy,{tf_x:test_x,tf_y:test_y})
        print('train loss: %.4f' %loss_,'| test accuracy: %.2f'%accuracy_)

test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')


















