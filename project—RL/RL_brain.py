
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,   #输出多少个actions对应Q值
            n_features,  #接受多少个observation的值，可以看做S的维数,描述该状态需要的向量的维数大小
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.95,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True,
            point_features=1,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter  #更新target network的间隔次数
        self.memory_size = memory_size
        self.batch_size = batch_size  #控制经验回放数据量
        self.epsilon_increment = e_greedy_increment    #对epsilon的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.point_features = point_features

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2)) #shape，行列。n_features * 2 + 2存储了两个状态以及上次状态采取的动作与获得的奖励

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("D:\数据融合组\project—RL\logs", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):


        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, 50700], name='s')  # input picture 4*50700
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        '''
        先把单独点放到s整体里来看
        '''
        self.points = tf.placeholder(tf.float32, [None, 2], name='points') #7*2

        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [50700, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                w22 = tf.get_variable('w22', [2, self.n_actions], initializer=w_initializer, collections=c_names)
                b22 = tf.get_variable('b22', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                self.q_eval = tf.matmul(l1, w2) + b2+tf.nn.relu(tf.matmul(self.points, w22) + b22)

            # first layer. collections is used later when assign to target net
            # with tf.variable_scope('l1'):
            #     w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
            #     b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
            #     l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            #
            # # second layer. collections is used later when assign to target net
            # with tf.variable_scope('l2'):
            #     w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
            #     b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
            #     self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, 50700], name='s_')
        self.points_ = tf.placeholder(tf.float32, [None, 2], name='points')

        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [50700, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                w22 = tf.get_variable('w22', [2, self.n_actions], initializer=w_initializer,
                                      collections=c_names)
                b22 = tf.get_variable('b22', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                self.q_next = tf.matmul(l1, w2) + b2 + tf.nn.relu(tf.matmul(self.points_, w22) + b22)

        # with tf.variable_scope('target_net'):
        #         # c_names(collections_names) are the collections to store variables
        #         c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        #
        #         # first layer. collections is used later when assign to target net
        #         with tf.variable_scope('l1'):
        #             w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
        #             b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
        #             l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
        #
        #         # second layer. collections is used later when assign to target net
        #         with tf.variable_scope('l2'):
        #             w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
        #             b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
        #             self.q_next = tf.matmul(l1, w2) + b2

            # c_names(collections_names) are the collections to store variables
            # c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            #
            # # first layer. collections is used later when assign to target net
            # with tf.variable_scope('l1'):
            #     w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
            #     b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
            #     l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
            #
            # # second layer. collections is used later when assign to target net
            # with tf.variable_scope('l2'):
            #     w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
            #     b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
            #     self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # transition = np.hstack((s, [a, r], s_))
        #
        # # replace the old memory with new memory
        # index = self.memory_counter % self.memory_size
        # self.memory[index, :] = transition
        #
        # self.memory_counter += 1

        transition = np.array((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1


    def choose_action(self, observation):        #传递数据的接口
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions

            '''
            错误：
            需要把原来的序列转成矩阵，现在data_s（是个序列）维数是（4，），想办法转为（4,50700）矩阵形式
            '''
            data_s=np.array(observation[0,[4,5,6,7]])
            # print(data_s)
            f_data_s=[]
            for i in data_s:
                f_data_s.append(i[0])
            f_data_s=np.array(f_data_s)  #维数化为（4,50700）
            # print(f_data_s.shape)

            data_points=np.array(observation[0,[0,1,2,3,8,9,10]])
            f_point_s = []
            for i in data_points:
                f_point_s.append([i[0], i[1]])
            f_point_s = np.array(f_point_s)  #维数化为了（7,2）
            #print(data_points.shape)

            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: f_data_s, self.points: f_point_s})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        '''转换输入数据的维数'''
        batch_memory_s_ = []
        for i in batch_memory[:, 2][4:8]:
            batch_memory_s_.append(i[0])
            batch_memory_s_ = np.array(batch_memory_s_)

        batch_memory_s = []
        for i in batch_memory[:, 0][4:8]:
            batch_memory_s.append(i[0])
            batch_memory_s = np.array(batch_memory_s)

        batch_memory_points_ = []
        for i in batch_memory[:, 2][0,1,2,3,8,9,10]:
            batch_memory_points_.append([i[0], i[1]])
        batch_memory_points_ = np.array(batch_memory_points_)

        batch_memory_points = []
        for i in batch_memory[:, 0][0, 1, 2, 3, 8, 9, 10]:
            batch_memory_points.append([i[0], i[1]])
        batch_memory_points = np.array(batch_memory_points)

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory_s_,
                self.points_:batch_memory_points_,# fixed params
                self.s: batch_memory_s,
                self.points: batch_memory_points,  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, 1][0].astype(int)
        reward = batch_memory[:, 1][1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        batch_memory_points = []
        for i in batch_memory[:, 0][0, 1, 2, 3, 8, 9, 10]:
            batch_memory_points.append([i[0], i[1]])
        batch_memory_points = np.array(batch_memory_points)

        batch_memory_s = []
        for i in batch_memory[:, 0][4:8]:
            batch_memory_s.append(i[0])
            batch_memory_s = np.array(batch_memory_s)



        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={
                                                self.s: batch_memory_s,
                                                self.points: batch_memory_points,
                                                self.q_target: q_target})

        print(str(self.cost))
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

# if __name__ == '__main__':
#     DQN = DeepQNetwork(49,3136, output_graph=True)

