#-*-coding:utf-8-*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


# 准备数据
digits=load_digits()
x=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
# x_train为训练数据，x_test为测试数据
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3)

# 定义添加层函数
# layer_name 代表其每层的名称
def add_layer(input,in_size,out_size,layer_name,activation_function=None):
    # 生成初始参数时，随机变量(normal distribution)会比全部为0要好很多
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases的推荐值不为0
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # tf.matmul()是矩阵的乘法
    out_pre = tf.matmul(input, weights) + biases
    # 调用dropout函数
    out_pre=tf.nn.dropout(out_pre,keep_prob)
    # 当activation_function——激励函数为None时，输出就是当前的预测值
    if activation_function is None:
        outputs = out_pre
    # 不为None时，就把Wx_plus_b传到activation_function()函数中得到输出
    else:
        outputs = activation_function(out_pre)
    # 绘制outputs图
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

# 构建所需的数据
# keep_prob定义保留概率，即要保留的结果所占比例，为一个placeholder，在run时传入
# 当keep_prob=1的时候，相当于100%保留，也就是dropout没有起作用。
keep_prob=tf.placeholder(tf.float32)
xs=tf.placeholder(tf.float32,[None,64]) #8*8=64个特征
ys=tf.placeholder(tf.float32,[None,10])

# 添加隐含层
l1 = add_layer(xs, 64, 50, 'L1', activation_function=tf.nn.tanh)
# 添加输出层
prediction=add_layer(l1,50,10,'L2',activation_function=tf.nn.softmax)
# loss函数选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，交叉熵就等于零
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
# 绘制loss变化图
tf.summary.scalar('loss', cross_entropy)
# 训练方法使用梯度下降法调优
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 变量初始化
sess=tf.Session()

# 所有训练图合并
# tf.merge_all_summaries() 方法会对所有的 summaries 合并到一起
# 使用tf.summary.FileWriter() 将绘画出的图保存到一个目录中
# 第二个参数需要使用sess.graph，将前面定义的框架信息收集起来，然后放在logs/目录下面
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

sess.run(tf.global_variables_initializer())

# 训练
# 机器学习的内容是train_step，用session来run每次training的数据
# 当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。
for i in range(500):  #500次训练
    # sess.run(train_step,feed_dict={xs:x_train,ys:y_train,keep_prob:1})
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
    if i%50==0:
        # merged 也需要run 才能发挥作用
        train_result = sess.run(merged, feed_dict={xs: x_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: x_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
        print(print(sess.run(cross_entropy,feed_dict={xs:x_train,ys:y_train})))