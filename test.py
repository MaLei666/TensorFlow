# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data
'''
初始化所有变量，进行训练

变量定义：
1. 生成x_data矩阵数据
2. 定义weights和biases变量参数

训练过程：
1. 计算y_data和y
    y_data为x_data生成的定值，为期望值
    生成一个weights和biases，计算y，为预测值
2. 计算y_data和y的误差loss
3. 调用梯度下降法调优，反向传递误差
    weights和biases参数更新
4. 定义训练次数
    通过每次训练、调优使loss达到最小误差
    
'''
# 创建数据
x_data = np.random.rand(100).astype(np.float32)  #创建随机100个数字定义为float32（比64节约内存，降低训练复杂度）1*100矩阵
y_data = x_data*0.1 + 0.3   #1*100矩阵

# 用 tf.Variable 来创建描述 y 的参数
# 搭建模型，构造variable的实例添加变量，构造函数需要变量的初始值，可以是任何类型和形状的tensor
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  #random_uniform返回1维矩阵，介于-1到1范围内，产生的值是均匀分布的
# Weights = tf.Variable(tf.zeros([1]))  #random_uniform返回1维矩阵，介于-1到1范围内，产生的值是均匀分布的
biases = tf.Variable(tf.zeros([1]))    #zeros输出1维矩阵，元素均为0

y = Weights*x_data + biases   #1*100矩阵

# 计算 y 和 y_data 的误差
loss = tf.reduce_mean(tf.square(y-y_data))   # 矩阵中每个元素求平方，reduce_mean计算元素平均值

# 使用梯度下降法GradientDescent反向传递误差，梯度下降法会使用全部样本
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 学习率learning_rate，过大导致震荡，无法得到最优解，过小导致学习过程漫长

# 使用 optimizer 进行参数的更新，使loss达到最小误差
train = optimizer.minimize(loss)

# 初始化所有之前定义的计算图中global variable的 op
init = tf.global_variables_initializer()  #将所有全局变量的初始化器汇总，并对其进行初始化
sess = tf.Session()      # 创建会话 Session
sess.run(init)          # 初始化参数

#  用 Session 来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性
for step in range(201):  #训练200次
    sess.run(train)         #进行训练
    if step % 20 == 0:      #每20次输出一个weights和biases状态
        print(step, sess.run(Weights), sess.run(biases),sess.run(y),y_data,sess.run(loss))