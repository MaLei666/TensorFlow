# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义添加层函数
def add_layer(input,in_size,out_size,activation_function=None):
    # 生成初始参数时，随机变量(normal distribution)会比全部为0要好很多
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    # biases的推荐值不为0
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    # tf.matmul()是矩阵的乘法
    out_pre=tf.matmul(input,weights)+biases
    # 当activation_function——激励函数为None时，输出就是当前的预测值
    if activation_function is None:
        outputs=out_pre
    # 不为None时，就把Wx_plus_b传到activation_function()函数中得到输出
    else:
        outputs=activation_function(out_pre)
    return outputs

# 构建所需的数据
x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

# 利用占位符定义我们所需的神经网络的输入
# None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

# 定义神经层
# 定义隐藏层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
# 定义输出层
prediction=add_layer(l1,10,1,activation_function=None)
# 计算预测和真实值的误差，差的平方求和再取平均
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
# 梯度下降法调优
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 变量初始化
sess=tf.Session()
sess.run(tf.global_variables_initializer())

# 散点图描述真实数据
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
# plt.ion()用于连续显示
plt.ion()
plt.show()

# 训练
# 机器学习的内容是train_step，用session来run每次training的数据
# 当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 ==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        # 显示预测数据
        # 每隔50次训练刷新一次图形
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # 用红色，宽度为5的线显示预测和输入数据之间的关系
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        # 暂停0.1s
        plt.pause(1)
# 绘图保持打开状态
plt.ioff()
plt.show()