import numpy as np
import time
import tensorflow as tf
import sys
import map_cut
from map_cut import ins2npy
import pandas as pd

rewdata = './insdata'

class TEST():
    def __init__(self):
        self.action_space = ['11', '12', '13', '14', '15', '16', '17', '21', '22', '23', '24', '25', '26', '27', '31',
                             '32', '33', '34', '35', '36', '37', '41', '42', '43', '44', '45', '46', '47', '51', '52',
                             '53', '54', '55', '56', '57', '61', '62', '63', '64', '65', '66', '67', '71', '72', '73',
                             '74', '75', '76', '77'
                             ]
        self.n_actions = len(self.action_space)
        self.action_times = 0  # 用于记录采取了多少次动作
        self.n_features = 11  # 该惯导点T4、以及前三个惯导点T1\T2\T3，四个惯导点周围所截取图片P1\P2\P3\P4，前三个决策出来的点D1\D2\D3
        # state [T1 T2 T3 T4\P1 P2 P3 P4\D1 D2 D3]

        self.T1 = np.array((0, 0))  # ins点
        self.T2 = np.array((0, 0))
        self.T3 = np.array((0, 0))
        self.T4 = np.array((0, 0))
        self.P1 = np.array(0)  # 图片信息
        self.P2 = np.array(0)
        self.P3 = np.array(0)
        self.P4 = np.array(0)
        self.D1 = np.array((0, 0))  # 初始状态让label点充当前面的决策点
        self.D2 = np.array((0, 0))
        self.D3 = np.array((0, 0))

        self.D4 = np.array((0, 0))  # 采取动作后得到的决策点
        self.acc = 0
        self.righttime = 0

        self.ins_num=0 #用第一个作为测试集
        # self._build_maze(num=self.ins_num)



    def _build_maze(self, num):
        # 得到所有惯导点附近图片、惯导点、label点（单位：像素）,保存下来 npy
        # 以dataframe形式保存，图片与各个点编号对应
        print("this is build_maze，the pic_cut code is running~")
        since = time.time()
        ins2npy(rewdata, num)
        time_elapsed = time.time() - since
        # 代码计时
        print('The pic_cut code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def reset(self):

        # 上一个惯导轨迹结束，重新输入此惯导轨迹需要此函数reset,返回observation（就是当前状态，7个点与4张图），即初始状态
        # 对于重新输入的看法：每一次因为predict点不同，对于决策来说S也不同，符合强化学习要求
        npydata = np.load("D:\数据融合组\project_RL_v2\insdata2npy\machine " + str(self.ins_num) + ".npy")  # 加载后列名变为0,1,2,3.....
        insdataframe = pd.DataFrame(npydata)
        self.action_times = 0
        self.acc = 0
        self.righttime = 0

        self.T1 = np.array((insdataframe.loc[1, 0], insdataframe.loc[1, 1]))  # ins点
        self.T2 = np.array((insdataframe.loc[2, 0], insdataframe.loc[2, 1]))
        self.T3 = np.array((insdataframe.loc[3, 0], insdataframe.loc[3, 1]))
        self.T4 = np.array((insdataframe.loc[4, 0], insdataframe.loc[4, 1]))
        self.P1 = np.array(insdataframe.loc[0, 5])  # 图片信息,每个图片是（1,50700）
        self.P2 = np.array(insdataframe.loc[1, 5])
        self.P3 = np.array(insdataframe.loc[2, 5])
        self.P4 = np.array(insdataframe.loc[3, 5])
        self.D1 = self.T1  # 测试中初始状态让惯导点充当前面的决策点
        self.D2 = self.T2
        self.D3 = self.T3
        self.observation = np.array(
            (self.T1, self.T2, self.T3, self.T4, self.P1, self.P2, self.P3, self.P4, self.D1, self.D2, self.D3))

        return self.observation

    def step(self, action):
        # #当前状态s [P1\P2\P3\P4\T1\T2\T3\T4\D1\D2\D3]
        # #采取动作，下一状态s_
        # # reward function
        # #done完成标识符
        # return s_, reward, done
        npydata = np.load("D:\数据融合组\project_RL_v2\insdata2npy\machine " + str(self.ins_num) + ".npy")  # 加载后列名变为0,1,2,3.....
        insdataframe = pd.DataFrame(npydata)

        self.action_times += 1
        take_action = int(self.action_space[action])
        # 动作上的单位1代表25个像素，惯导中心点坐标为44
        action_x = int(take_action / 10)
        action_y = int(take_action % 10)

        delta_y = (action_x - 4) * 25
        delta_x = (action_y - 4) * 25

        now_loc = self.T4
        self.D4 = np.array([now_loc[0] + delta_x, now_loc[1] + delta_y])  # 本次作出的决策点

        self.T1 = self.T2  # ins点
        self.T2 = self.T3
        self.T3 = self.T4
        self.T4 = np.array((insdataframe.loc[4 + self.action_times, 0], insdataframe.loc[4 + self.action_times, 1]))
        self.P1 = self.P2  # 图片信息
        self.P2 = self.P3
        self.P3 = self.P4
        self.P4 = np.array(insdataframe.loc[3 + self.action_times, 5])
        self.D1 = self.D2  # 初始状态让label点充当前面的决策点
        self.D2 = self.D3
        self.D3 = self.D4

        self.observation = np.array(
            (self.T1, self.T2, self.T3, self.T4, self.P1, self.P2, self.P3, self.P4, self.D1, self.D2, self.D3))

        # reward function
        dec2label_distance = np.sqrt((self.D4[0] - insdataframe.loc[3 + self.action_times, 2]) ** 2 + (
                    self.D4[1] - insdataframe.loc[3 + self.action_times, 3]) ** 2)
        # print("dec2label_distance "+str(dec2label_distance))
        t2label_distance = insdataframe.loc[3 + self.action_times, 4]
        # print("t2label_distance " + str(t2label_distance))

        if dec2label_distance == 0:
            dec2label_distance = 1
            self.righttime += 1

        if dec2label_distance < t2label_distance:
            reward = 0
            self.righttime += 1
            # print(self.righttime)
        else:
            reward = 0

        # 判断一次轨迹是否完成
        done = False
        if (5 + self.action_times) == len(insdataframe):
            done = True
            self.acc = self.righttime / self.action_times
            print("test____acc ： " + str(self.acc))

        return self.observation, reward, done

class DeepQNetwork_test:
    def __init__(
            self,
            n_actions,   #输出多少个actions对应Q值
            n_features,  #接受多少个observation的值，可以看做S的维数,描述该状态需要的向量的维数大小
            e_greedy=1.0,
            point_features=1,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.epsilon_max = e_greedy
        self.point_features = point_features

        # total learning step
        self.learn_step_counter = 0

        # consist of [target_net, evaluate_net]
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, 50700], name='s')  # input picture 4*50700
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.points = tf.placeholder(tf.float32, [None, 2], name='points')  # 输入惯导与决策点 7*2
        self.keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(self.s, [-1, 130, 130, 3])



        def conv2d(x, W):
            # stride [1, x_movement, y_movement, 1]
            # Must have strides[0] = strides[3] = 1
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            # stride [1, x_movement, y_movement, 1]
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [5, 5, 3, 64], collections=c_names)
                b1 = tf.get_variable('b1', [64], collections=c_names)
                l1 = tf.nn.relu(conv2d(x_image, w1) + b1)  # out 130*130*64
                h_pool1 = max_pool_2x2(l1)  # out 65*65*64

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [5, 5, 64, 128],  collections=c_names)
                b2 = tf.get_variable('b2', [128], collections=c_names)
                l2 = tf.nn.relu(conv2d(h_pool1, w2) + b2)  # 65*65*128
                h_pool2 = max_pool_2x2(l2)  # out 33*33*128    right???

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [33 * 33 * 128, 1024],  collections=c_names)
                b3 = tf.get_variable('b3', [1024],  collections=c_names)
                h_pool2 = tf.reshape(h_pool2, [-1, 33 * 33 * 128])
                fc1 = tf.nn.relu(tf.matmul(h_pool2, w3) + b3)
                # fc1 = tf.nn.dropout(fc1, self.keep_prob)  # num*1024

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [1024, self.n_actions],  collections=c_names)
                b4 = tf.get_variable('b4', [self.n_actions], collections=c_names)
                fc2 = tf.nn.relu(tf.matmul(fc1, w4) + b4)  # 4*49

                w22 = tf.get_variable('w22', [2, self.n_actions], collections=c_names)
                b22 = tf.get_variable('b22', [1, self.n_actions], collections=c_names)

                q_eval_2 = tf.nn.relu(tf.matmul(self.points, w22) + b22)  # 7*49

                q_eval_1 = tf.reshape(fc2, [-1, 196])  # 196=4*49
                q_eval_2 = tf.reshape(q_eval_2, [-1, 343])  # 343=7*49

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [196, self.n_actions],  collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions],  collections=c_names)

                w52 = tf.get_variable('w52', [343, self.n_actions],  collections=c_names)

                self.q_eval = tf.matmul(q_eval_1, w5) + b5 + tf.matmul(q_eval_2, w52)

    def choose_action(self, observation):        #传递数据的接口
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]


        data_s=np.array(observation[0,[4,5,6,7]])
        f_data_s=[]
        for i in data_s:
            f_data_s.append(i[0])
        f_data_s=np.array(f_data_s)  #维数化为（4,50700）

        data_points=np.array(observation[0,[0,1,2,3,8,9,10]])
        f_point_s = []
        for i in data_points:
            f_point_s.append([i[0], i[1]])
        f_point_s = np.array(f_point_s)  #维数化为了（7,2）

        model_file = tf.train.latest_checkpoint("./my_model/")

        self.saver = tf.train.Saver()

        self.saver.restore(self.sess, model_file)
        # print(self.sess.run(w52))

        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: f_data_s, self.points: f_point_s,self.keep_prob:1.0})
        # print(self.sess.run(w1))
        action = np.argmax(actions_value)

        return action

def test_acc():
    env=TEST()
    step = 0
    print("---------------- test begin !!!------------------------" )

    observation = env.reset()
    RL_test=DeepQNetwork_test(env.n_actions, env.n_features,
                      e_greedy=1.0,
                      )

    while True:
        # RL choose action based on observation

        # observation = np.array((self.T1, self.T2, self.T3, self.T4, self.P1, self.P2, self.P3, self.P4, self.D1, self.D2, self.D3))
        action = RL_test.choose_action(observation)

        # RL take action and get next observation and reward
        observation_, reward, done = env.step(action)

        # swap observation
        observation = observation_

        # break while loop when end of this episode
        if done:
            break
        step += 1


if __name__=="__main__":
    test_acc()

