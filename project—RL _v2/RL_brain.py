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
        self.memory = np.zeros((self.memory_size, 3),dtype=object) #shape，3 存储了两个状态以及上次状态采取的动作与获得的奖励[s,[a,r],s_]

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("D:\数据融合组\project_RL_v2\logs ", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.train.Saver(max_to_keep=1)
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, 50700], name='s')  # input picture 4*50700
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        self.points = tf.placeholder(tf.float32, [None, 2], name='points')  # 输入惯导与决策点 7*2
        self.keep_prob = tf.placeholder(tf.float32,name='prob')

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
                w1 = tf.get_variable('w1', [5, 5, 3, 64], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [64], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(conv2d(x_image, w1) + b1)  # out 130*130*64
                h_pool1 = max_pool_2x2(l1)  # out 65*65*64

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [5, 5, 64, 128], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [128], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(conv2d(h_pool1, w2) + b2)  # 65*65*128
                h_pool2 = max_pool_2x2(l2)  # out 33*33*128    right???

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [33 * 33 * 128, 1024], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1024], initializer=b_initializer, collections=c_names)
                h_pool2 = tf.reshape(h_pool2, [-1, 33 * 33 * 128])
                fc1 = tf.nn.relu(tf.matmul(h_pool2, w3) + b3)
                fc1 = tf.nn.dropout(fc1, self.keep_prob)  # num*1024

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [1024, self.n_actions], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [self.n_actions], initializer=b_initializer, collections=c_names)
                fc2 = tf.nn.relu(tf.matmul(fc1, w4) + b4)  # 4*49

                w22 = tf.get_variable('w22', [2, self.n_actions], initializer=w_initializer, collections=c_names)
                b22 = tf.get_variable('b22', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                q_eval_2 = tf.nn.relu(tf.matmul(self.points, w22) + b22)  # 7*49

                q_eval_1 = tf.reshape(fc2, [-1, 196])  # 196=4*49
                q_eval_2 = tf.reshape(q_eval_2, [-1, 343])  # 343=7*49

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [196, self.n_actions], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                w52 = tf.get_variable('w52', [343, self.n_actions], initializer=w_initializer, collections=c_names)

                self.q_eval = tf.add(tf.add(tf.matmul(q_eval_1, w5) , b5) , tf.matmul(q_eval_2, w52), name="Q_eval")

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            # self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, 50700], name='s_')
        self.points_ = tf.placeholder(tf.float32, [None, 2], name='points')

        with tf.variable_scope('target_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [5, 5, 3, 64], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [64], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(conv2d(x_image, w1) + b1)  # out 130*130*64
                h_pool1 = max_pool_2x2(l1)  # out 65*65*64

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [5, 5, 64, 128], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [128], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(conv2d(h_pool1, w2) + b2)  # 65*65*128
                h_pool2 = max_pool_2x2(l2)  # out 33*33*128    right???

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [33 * 33 * 128, 1024], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1024], initializer=b_initializer, collections=c_names)
                h_pool2 = tf.reshape(h_pool2, [-1, 33 * 33 * 128])
                fc1 = tf.nn.relu(tf.matmul(h_pool2, w3) + b3)
                fc1 = tf.nn.dropout(fc1, self.keep_prob)  # num*1024

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [1024, self.n_actions], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [self.n_actions], initializer=b_initializer, collections=c_names)
                fc2 = tf.nn.relu(tf.matmul(fc1, w4) + b4)  # 4*49

                w22 = tf.get_variable('w22', [2, self.n_actions], initializer=w_initializer, collections=c_names)
                b22 = tf.get_variable('b22', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                q_eval_2 = tf.nn.relu(tf.matmul(self.points_, w22) + b22)  # 7*49

                q_next_1 = tf.reshape(fc2, [-1, 196])  # 196=4*49
                q_next_2 = tf.reshape(q_eval_2, [-1, 343])  # 343=7*49

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [196, self.n_actions], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                w52 = tf.get_variable('w52', [343, self.n_actions], initializer=w_initializer, collections=c_names)

                self.q_next = tf.matmul(q_next_1, w5) + b5 + tf.matmul(q_next_2, w52)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # transition = np.hstack((s, [a, r], s_))
        # # replace the old memory with new memory
        # index = self.memory_counter % self.memory_size
        # self.memory[index, :] = transition
        #
        # self.memory_counter += 1
        s=np.array(s)
        s_ = np.array(s_)
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
            解决数据结构不匹配问题，后面操作类似
            '''
            data_s=np.array(observation[0,[4,5,6,7]])
            # print(data_s)
            f_data_s=[]
            for i in data_s:
                f_data_s.append(i[0])
            f_data_s=np.array(f_data_s)  #维数化为（4,50700）

            data_points=np.array(observation[0,[0,1,2,3,8,9,10]])
            f_point_s = []
            for i in data_points:
                f_point_s.append([i[0], i[1]])
            f_point_s = np.array(f_point_s)  #维数化为了（7,2）

            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: f_data_s, self.points: f_point_s,self.keep_prob:0.5})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):

        self.saver = tf.train.Saver(max_to_keep=1)
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

        batch_memory_s_=np.array([])
        t = 0
        for observation in batch_memory[:, 2]:#[0,[4,5,6,7]]
            observation=np.array([observation])
            data_s = np.array(observation[0,[4, 5, 6, 7]])

            f_data_s = []
            for i in data_s:
                f_data_s.append(i[0])
            f_data_s=np.array(f_data_s)

            if t==0:
                batch_memory_s_=f_data_s
            if t>0:
                batch_memory_s_ = np.concatenate((batch_memory_s_,f_data_s))
            t+=1
        batch_memory_s_ = np.array(batch_memory_s_)

        batch_memory_s = np.array([])
        t = 0
        for observation in batch_memory[:, 0]:  # [0,[4,5,6,7]]

            observation = np.array([observation])
            data_s = np.array(observation[0, [4, 5, 6, 7]])

            f_data_s = []
            for i in data_s:
                f_data_s.append(i[0])
            f_data_s = np.array(f_data_s)

            if t == 0:
                batch_memory_s = f_data_s
            if t > 0:
                batch_memory_s = np.concatenate((batch_memory_s, f_data_s))
            t += 1
        batch_memory_s = np.array(batch_memory_s)


        batch_memory_points_ = np.array([])
        t = 0
        for observation in batch_memory[:, 2]:  # [0,[4,5,6,7]]

            observation = np.array([observation])
            data_s = np.array(observation[0, [0, 1, 2, 3, 8, 9, 10]])

            f_data_s = []
            for i in data_s:
                f_data_s.append([i[0], i[1]])
            f_data_s = np.array(f_data_s)

            if t == 0:
                batch_memory_points_ = f_data_s
            if t > 0:
                batch_memory_points_ = np.concatenate((batch_memory_points_, f_data_s))
            t += 1
            batch_memory_points_ = np.array(batch_memory_points_)

        batch_memory_points = np.array([])
        t = 0
        for observation in batch_memory[:, 0]:  # [0,[4,5,6,7]]

            observation = np.array([observation])
            data_s = np.array(observation[0, [0, 1, 2, 3, 8, 9, 10]])
            f_data_s = []
            for i in data_s:
                f_data_s.append([i[0], i[1]])
            f_data_s = np.array(f_data_s)

            if t == 0:
                batch_memory_points = f_data_s
            if t > 0:
                batch_memory_points = np.concatenate((batch_memory_points, f_data_s))
            t += 1
            batch_memory_points = np.array(batch_memory_points)


        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory_s_,
                self.points_:batch_memory_points_,
                self.keep_prob: 0.5,
                self.s: batch_memory_s,
                self.points: batch_memory_points,

            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        list_memory=batch_memory[:, 1][0]

        eval_act_index =int(list_memory[0])
        reward =list_memory[1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network

        batch_memory_points = np.array([])
        t = 0
        for observation in batch_memory[:, 0]:  # [0,[4,5,6,7]]

            observation = np.array([observation])
            data_s = np.array(observation[0, [0, 1, 2, 3, 8, 9, 10]])

            f_data_s = []
            for i in data_s:
                f_data_s.append([i[0], i[1]])
            f_data_s = np.array(f_data_s)

            if t == 0:
                batch_memory_points = f_data_s
            if t > 0:
                batch_memory_points = np.concatenate((batch_memory_points, f_data_s))
            t += 1
            batch_memory_points = np.array(batch_memory_points)

        batch_memory_s = np.array([])
        t = 0
        for observation in batch_memory[:, 0]:  # [0,[4,5,6,7]]

            observation = np.array([observation])
            data_s = np.array(observation[0, [4, 5, 6, 7]])

            f_data_s = []
            for i in data_s:
                f_data_s.append(i[0])
            f_data_s = np.array(f_data_s)

            if t == 0:
                batch_memory_s = f_data_s
            if t > 0:
                batch_memory_s = np.concatenate((batch_memory_s, f_data_s))
            t += 1
        batch_memory_s = np.array(batch_memory_s)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={
                                                self.s: batch_memory_s,
                                                self.points: batch_memory_points,
                                                self.q_target: q_target,
                                                self.keep_prob: 0.5
                                                })

        # print(str(self.cost))
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        model_path = "./my_model/model.ckpt"

        self.saver.save(self.sess, model_path)
        # print(str(save_path))

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

# if __name__ == '__main__':
#     DQN = DeepQNetwork(49,3136, output_graph=True)

