import numpy as np
import time
import sys
from map_cut import Pic2Vec


# UNIT = 40   # pixels
# MAZE_H = 4  # grid height
# MAZE_W = 4  # grid width

'''
如何对应动作序号和每个点的关系？挨个键入或者for循环计算对应？
'''

class Maze():
    def __init__(self):
        self.action_space = ['11','12','13','14','15','16','17','21','22','23','24','25','26','27','31','32','33','34','35','36','37','41','42','43','44','45','46','47','51','52','53','54','55','56','57','61','62','63','64','65','66','67','71','72','73','74','75','76','77'
]
        self.n_actions = len(self.action_space)
        self.n_features = 2*7+4*28*28 #该惯导点T4、以及前三个惯导点T1\T2\T3，四个惯导点周围所截取图片P1\P2\P3\P4，前三个决策出来的点D1\D2\D3
        self._build_maze()

    def _build_maze(self):
        #得到所有惯导点附近图片、惯导点、label点（单位：像素）,保存下来
        #以dataframe形式保存，图片与各个点编号对应

    def reset(self,track_):

        #上一个惯导轨迹结束，需要输入下一个惯导轨迹需要此函数reset,返回observation（就是当前状态，7个点与4张图）

    def step(self, action):
        #当前状态s [P1\P2\P3\P4\T1\T2\T3\T4\D1\D2\D3]
        #采取动作，下一状态s_
        # reward function
        #done完成标识符
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        # self.update()