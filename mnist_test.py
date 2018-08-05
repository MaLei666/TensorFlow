#-*-coding:utf-8-*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# 导入MNIST手写数字库
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
# 定义添加层函数
def add_layer(input,in_size,out_size,activation_function=None):
    # 生成初始参数时，随机变量(normal distribution)会比全部为0要好很多
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases的推荐值不为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # tf.matmul()是矩阵的乘法
    out_pre = tf.matmul(input, weights) + biases
    # 当activation_function——激励函数为None时，输出就是当前的预测值
    if activation_function is None:
        outputs = out_pre
    # 不为None时，就把Wx_plus_b传到activation_function()函数中得到输出
    else:
        outputs = activation_function(out_pre)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# 构建所需的数据
xs=tf.placeholder(tf.float32,[None,784]) #28*28=784个特征
# 输出数字为0-9，共10类
ys=tf.placeholder(tf.float32,[None,10])
# 定义输出层
prediction=add_layer(xs,784,10,activation_function=tf.nn.softmax)
# loss函数选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，交叉熵就等于零
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
# 训练方法使用梯度下降法调优
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 变量初始化
sess=tf.Session()
sess.run(tf.global_variables_initializer())

# 训练
# 机器学习的内容是train_step，用session来run每次training的数据
# 当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
