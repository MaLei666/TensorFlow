# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# 参数定义
lr=0.02                 # 学习率
cell_size=32            # rnn cell size
inputs_size=1           # rnn 输入
time_steps=10           # 数据高度

# 生成数据
# 在指定的间隔[start,stop]内返回均匀间隔的数字。
steps=np.linspace(0,np.pi*2,100,dtype=np.float32)
# sin、cos函数，steps为其角度，单位为弧度
x_np=np.sin(steps)
y_np=np.cos(steps)
# 绘制图像，‘r-’为格式化字符串[color][marker][line]
# steps,y_np为点的坐标
plt.plot(steps,y_np,'r-',label='target(cos)')
plt.plot(steps,x_np,'b-',label='input(sin)')
# 图例摆放位置，默认右上方
plt.legend(loc='best')
plt.show()

# 设置TensorFlow占位符，dtype=float32，shape=(batch, 10, 1)
tf_x=tf.placeholder(tf.float32,[None,time_steps,inputs_size])
tf_y=tf.placeholder(tf.float32,[None,time_steps,inputs_size])

# 定义rnn主体结构
# cell定义,num_units：int，RNN单元中的单元数
rnn_cell=tf.contrib.rnn.BasicRNNCell(num_units=cell_size)
# 返回零填充状态张量
init_state=rnn_cell.zero_state(batch_size=1,dtype=tf.float32)

# 创建由RNNCell指定的递归神经网络cell
# 函数中的 time_major 参数会针对不同 inputs 格式有不同的值.
outputs,final_s=tf.nn.dynamic_rnn(  # 输出output和state(最后的状态)
    rnn_cell,            #RNNCell的一个实例
    tf_x,           # RNN输入
    initial_state=init_state,  #RNN的初始状态
    time_major=False # inputs和outputs张量的形状格式。False: (batch, time step, input); True: (time step, batch, input)
)
# 给定tensor，返回tensor与形状具有相同值的张量shape
# -1使得计算该维度的大小，以使总大小保持不变
out2d=tf.reshape(outputs,[-1,cell_size])   # tensor=outputs，shape=[-1,cell_size]
# 输出与units大小相同张量
net_out2d=tf.layers.dense(out2d,inputs_size)    # 张量输入=out2d，输出空间维数=inputs_size
out=tf.reshape(net_out2d,[-1,time_steps,inputs_size])
# labels=真实输出，predictions=预测输出，求误差平方和
loss=tf.losses.mean_squared_error(labels=tf_y,predictions=out)
# 学习率=lr 要最小化的值=loss
train_op=tf.train.AdamOptimizer(lr).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
# 控制子图默认间距，figsize为整数元组，提供高度和宽度
plt.figure(1,figsize=(12,5),facecolor='r')
# 打开交互模式
plt.ion()

for step in range(60):
    start,end=step*np.pi,(step+1)*np.pi

    steps=np.linspace(start,end,time_steps)
    x=np.sin(steps)[np.newaxis,:,np.newaxis]
    y=np.cos(steps)[np.newaxis,:,np.newaxis]
    if 'final_s' not in globals():
        feed_dict={tf_x:x,tf_y:y}
    else:
        feed_dict={tf_x:x,tf_y:y,init_state:final_s}

    _,pred_,final_s_=sess.run([train_op,out,final_s],feed_dict)
    plt.plot(steps.flatten(),'r-')
    plt.plot(steps,pred_.flatten(),'b-')
    plt.ylim(-1.2,1.2)
    plt.draw()
    plt.pause(0.5)
plt.ioff()
plt.show()

















