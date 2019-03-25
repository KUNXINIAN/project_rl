
import numpy as np
import time
import sys
import map_cut
from map_cut import ins2npy
import pandas as pd

rewdata = './insdata'

class Maze():
    def __init__(self):
        self.action_space = ['11','12','13','14','15','16','17','21','22','23','24','25','26','27','31','32','33','34','35','36','37','41','42','43','44','45','46','47','51','52','53','54','55','56','57','61','62','63','64','65','66','67','71','72','73','74','75','76','77'
]
        self.n_actions = len(self.action_space)
        self.action_times=0   #用于记录采取了多少次动作
        self.n_features = 11 #该惯导点T4、以及前三个惯导点T1\T2\T3，四个惯导点周围所截取图片P1\P2\P3\P4，前三个决策出来的点D1\D2\D3
        #state [T1 T2 T3 T4\P1 P2 P3 P4\D1 D2 D3]

        self.T1 = np.array((0, 0) ) # ins点
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

        self.D4 = np.array((0, 0)) #采取动作后得到的决策点
        self.acc=0
        self.righttime=0

        #self._build_maze(num=2)

    def _build_maze(self,num):
        #得到所有惯导点附近图片、惯导点、label点（单位：像素）,保存下来 npy
        #以dataframe形式保存，图片与各个点编号对应
        print("this is build_maze，the pic_cut code is running~")
        since = time.time()
        ins2npy(rewdata, num)
        time_elapsed = time.time() - since
        #代码计时
        print('The pic_cut code run {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    def reset(self):

        #上一个惯导轨迹结束，重新输入此惯导轨迹需要此函数reset,返回observation（就是当前状态，7个点与4张图），即初始状态
        #对于重新输入的看法：每一次因为predict点不同，对于决策来说S也不同，符合强化学习要求
        npydata=np.load("D:\数据融合组\project_RL_v2\insdata2npy\machine "+str(2)+".npy") #加载后列名变为0,1,2,3.....
        insdataframe = pd.DataFrame(npydata)
        self.action_times=0
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
        self.D1 = np.array((insdataframe.loc[1, 2], insdataframe.loc[1, 3]))  # 初始状态让label点充当前面的决策点
        self.D2 = np.array((insdataframe.loc[2, 2], insdataframe.loc[2, 3]))
        self.D3 = np.array((insdataframe.loc[3, 2], insdataframe.loc[3, 3]))
        self.observation = np.array(
            (self.T1, self.T2, self.T3, self.T4, self.P1, self.P2, self.P3, self.P4, self.D1, self.D2, self.D3))

        return self.observation

    def step(self, action):
        # #当前状态s [P1\P2\P3\P4\T1\T2\T3\T4\D1\D2\D3]
        # #采取动作，下一状态s_
        # # reward function
        # #done完成标识符
        # return s_, reward, done
        npydata = np.load("D:\数据融合组\project_RL_v2\insdata2npy\machine " + str(2) + ".npy")  # 加载后列名变为0,1,2,3.....
        insdataframe = pd.DataFrame(npydata)

        self.action_times+=1
        take_action=int(self.action_space[action])
        #动作上的单位1代表25个像素，惯导中心点坐标为44
        action_x=int(take_action/10)
        action_y=int(take_action%10)

        delta_y = (action_x - 4) * 25
        delta_x = (action_y - 4) * 25

        now_loc=self.T4
        self.D4=np.array([now_loc[0]+delta_x,now_loc[1]+delta_y]) # 本次作出的决策点

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
        dec2label_distance=np.sqrt((self.D4[0]-insdataframe.loc[3+ self.action_times,2])**2+(self.D4[1]-insdataframe.loc[3+ self.action_times,3])**2)
        t2label_distance=insdataframe.loc[3+ self.action_times,4]

        if dec2label_distance == 0:
            dec2label_distance = 1
            self.righttime +=1

        if dec2label_distance < t2label_distance:
            reward = 100/dec2label_distance
            self.righttime += 1
        else:
            reward = -(dec2label_distance-t2label_distance)

        #判断一次轨迹是否完成
        done = False
        if (5 + self.action_times)==len(insdataframe):
            done = True
            self.acc=self.righttime/self.action_times
            print("train____acc ： "+str(self.acc))

        return self.observation, reward, done


    def render(self):
        pass

if __name__=="__main__":
    maze=Maze()
    print(maze.reset())
    print("after take action~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(maze.step(34))
    print("after take action~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(maze.step(22))