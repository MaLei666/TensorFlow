#-*-coding:utf-8-*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 导入MNIST手写数字库
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

# 定义weight变量，输入shape，返回变量的参数
# 使用tf.truncated_normal产生随机变量来初始化
def weight_variable(shape):
    inital=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)
# 使用tf.constant常量函数进行初始化
def bias_variable(shape):
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital)
# 定义卷积，tf.nn.conv2d函数是tensoflow里面的二维的卷积函数
# x是图片的所有参数，W是此卷积层的权重，然后定义步长strides=[1,1,1,1]
# strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步，这样得到的图片尺寸没有变化
# padding采用的方式是SAME。
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
# 定义池化，pooling 有两种，一种是最大值池化，一种是平均值池化，
# 采用最大值池化tf.max_pool()。池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 构建所需的数据
xs=tf.placeholder(tf.float32,[None,784])/255 #28*28=784个特征
# 输出数字为0-9，共10类
ys=tf.placeholder(tf.float32,[None,10])
# keep_prob定义保留概率，即要保留的结果所占比例，为一个placeholder，在run时传入
# 当keep_prob=1的时候，相当于100%保留，也就是dropout没有起作用。
keep_prob=tf.placeholder(tf.float32)

# 重定义输入数据大小，-1代表先不考虑输入的图片例子多少这个维度，后面的1是channel的数量，
# 图片是黑白的channel是1，RGB图像channel是3。
x_image=tf.reshape(xs,[-1,28,28,1])

# 建立卷积层
# 定义第一层卷积
# 定义weight，选择卷积核patch大小是5x5，黑白图片channel是1，输出是32个featuremap（即使用了32个卷积核）
w_conv1=weight_variable([5,5,1,32])
# 定义bias，大小是32个长度
b_conv1=bias_variable([32])
# 定义第一个卷积层h_conv1，同时对其进行非线性（激活函数tf.nn.relu修正线性单元）处理
# 因为采用了SAME的padding方式，输出图片的大小依然是28x28，只是厚度变厚，28*28*32
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
# 经过pooling处理后输出变为14*14*32
h_pool1=max_pool(h_conv1)

# 定义第二层卷积
# 本层输入即为第一层输出，卷积核设置为5*5，输入为32，输出定为64
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
# 定义第二个卷积层h_conv2，输出大小为14*14*64
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
# pooling处理后输出变为7*7*64
h_pool2=max_pool(h_conv2)

# 建立全连接层
# 建立全连接层1
# 将h_pool2通过reshape从三维变成一维数据
# -1表示先不考虑输入图片例子维度, 将结果展平
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
# 此时weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64，
# [n_samples,7,7,64]->>[n_samples,7*7*64]，后面的输出size定为1024
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
# h_pool2_flat与本层的W_fc1相乘（注意这个时候不是卷积了）
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
# 加入dropout处理过拟合问题
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

# 建立全连接层2
# 输入为1024，输出为10个类，prediction为预测值
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
# 使用softmax分类器对输出进行分类
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

# 优化
# loss函数选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，交叉熵就等于零
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
# 用tf.train.AdamOptimizer()作为优化器进行优化
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 变量初始化
sess=tf.Session()
sess.run(tf.global_variables_initializer())

# 训练
# 机器学习的内容是train_step，用session来run每次training的数据
# 当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%0==0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))



