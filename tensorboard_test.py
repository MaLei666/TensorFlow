# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 添加n_layer标识层数
def add_layer(input,in_size,out_size,n_layer,activation_function=None):
    # layer_name 代表其每层的名名称
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights=tf.Variable(tf.random_normal([in_size,out_size]),name='w')
            # tf.histogram_summary()方法用来绘制图片
            # 第一个参数是图表的名称, 第二个参数是图表要记录的变量
            tf.summary.histogram(layer_name + '/weights', weights)

        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            # 绘制biases图
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('out_pre'):
            out_pre=tf.matmul(input,weights)+biases
        if activation_function is None:
            outputs=out_pre
        else:
            outputs=activation_function(out_pre)
        # 绘制outputs图
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

# 构建所需的数据
x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

# with tf.name_scope()将xs和ys包含进来，形成一个大的图层，图层名字就是方法里的参数。
with tf.name_scope('inputs'):
    # 为xs指定名称为x_input，ys为y_input
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')

# 修改调用addlayer()函数的地方,添加n_layer参数
l1=add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,n_layer=2,activation_function=None)

# 绘制loss和变化图
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

# 绘制train
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 变量初始化
sess=tf.Session()

# 所有训练图合并
# tf.merge_all_summaries() 方法会对所有的 summaries 合并到一起
# 使用tf.summary.FileWriter() 将绘画出的图保存到一个目录中
# 第二个参数需要使用sess.graph，将前面定义的框架信息收集起来，然后放在logs/目录下面
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
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
        # merged 也需要run 才能发挥作用
        rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(rs, i)

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

# 到项目文件夹下，终端开启tensorboard
# tensorboard --logdir=logs
# 浏览器根据终端输出的网址，Chrome打开