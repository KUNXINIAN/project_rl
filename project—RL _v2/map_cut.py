import tensorflow as tf
from tensorflow.contrib import image
from tensorflow.python.ops import math_ops
import math
PI = math.pi

from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy import math

rewdata = './insdata'

# class get_data(object):

def FileList(rewdata):
        rewdata_path = os.path.abspath(rewdata)
        file_path = os.chdir(rewdata_path)
        fileListAll = os.listdir(file_path)  # 返回一个该路径下的文件名list
        filelist = []
        for file in fileListAll:
            filelist.append(file)
        return filelist

def read_data(rewdata_path,ins_num):
        '''
        需要能够读取CSV中的数据，
        ins_num 代表第几个csv, 0 代表第 1 个，以此类推
        返回一个dataframe
        形如：
        ins_x     ins_y  label_x  label_y  delta_ins2label
0    3128.78  1794.660   3133.0   1806.2        12.287392
1    3135.43  1758.300   3133.0   1769.4        11.362874

        '''
        data = pd.DataFrame()
        dir_list = FileList(rewdata_path)
        # for i in range(len(dir_list)):
        data_new = pd.read_csv(dir_list[ins_num],header=None, usecols=[0, 1, 3, 4])
        data = pd.concat([data, data_new], axis=0)
        data.index = range(len(data))
        data.columns = ['ins_x', 'ins_y', 'label_x', 'label_y']
        #对dataframe列操作计算方法如下
        data['delta_ins2label']=data.apply(lambda x:np.sqrt((x['ins_x']-x['label_x'])*(x['ins_x']-x['label_x'])+(x['ins_y']-x['label_y'])*(x['ins_y']-x['label_y'])),axis=1)

        return data

def ImageToMatrix(filename):
    '''
    修改为输入im,不是filename
    '''
    # 读取图片
    im = Image.open(filename)
    width,height = im.size
    print("width "+str(width)+" height "+str(height))
    im = im.convert("L")
    data = im.getdata()
    #data = np.matrix(data, dtype='float') / 1.0
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    print(new_data.shape)
    return new_data
#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def Pic2Vec(pic_x,pic_y):
    #输入坐标，直接得到向量
    im=mapCut(pic_x,pic_y)
    vec=ImageToMatrix(im)
    return vec

def getRotatePoint(map_shape, rotate_center, rotate_theta, origin_point):
    """
    实现功能，得到绕旋转中心旋转theta角度后的坐标
    :param map_shape:原始地图的尺寸，因为Image中的坐标原点在图片左上角，需要改变坐标系    Tensor-[height,width,channel]
    :param rotate_center:旋转中心   Tensor-[loc_x,loc_y]
    :param rotate_theta:旋转角度   Tensor-[theta]
    :param origin_point:需要进行旋转操作的点集 Tensor-[loc_x,loc_y]
    :return: rotate_point_list: Tensor-[loc_x,loc_y]
    """
    row = map_shape[0]
    center_x = rotate_center[0]
    center_y = row - rotate_center[1]
    point_x = origin_point[0]
    point_y = row - origin_point[1]

    after_rotate_x = math_ops.round(
        (point_x - center_x) * math_ops.cos(rotate_theta) - (point_y - center_y) * math_ops.sin(
            rotate_theta) + center_x)
    after_rotate_y = row - math_ops.round(
        (point_x - center_x) * math_ops.sin(rotate_theta) + (point_y - center_y) * math_ops.cos(
            rotate_theta) + center_y)
    rotate_point = [after_rotate_x, after_rotate_y]
    rotate_point = tf.reshape(rotate_point, [2])
    return rotate_point


def pointLegalCheck(map_shape, point, box_shape):
    """
    检测旋转后的点是否越界
    :param map_shape: 原始地图大小Tensor-[height,width,channel]
    :param point: 旋转后的点集Tensor-[4,2]
    :param box_shape: 裁剪图片区域的大小Tensor-[2]
    :return: 合法性检测后的点集
    """
    x_move_list = []
    y_move_list = []
    map_shape = tf.cast(map_shape, tf.float32)
    # 依次计算四个顶点与地图边界的偏移大小
    for i in range(4):
        x_move = tf.cond(tf.less(point[i][0], 0.0),
                         true_fn=lambda: 0.0 - point[i][0],  # 若坐标小于零，则需要向正方向移动
                         false_fn=lambda: tf.cond(tf.less(map_shape[1], point[i][0]),
                                                  # 若坐标大于地图边界，则需要向负方向移动
                                                  true_fn=lambda: map_shape[1] - point[i][0],
                                                  false_fn=lambda: 0.0  # 坐标在地图中无需移动
                                                  )
                         )
        y_move = tf.cond(tf.less(point[i][1], 0.0),
                         true_fn=lambda: 0.0 - point[i][1],
                         false_fn=lambda: tf.cond(tf.less(map_shape[0], point[i][1]),
                                                  true_fn=lambda: map_shape[0] - point[i][1],
                                                  false_fn=lambda: 0.0
                                                  )
                         )
        x_move_list.append(x_move)
        y_move_list.append(y_move)

    # 依据四个顶点的偏移，得到坐标的修正值
    x_move = getPixelMove(tf.reshape(x_move_list, [4]))
    y_move = getPixelMove(tf.reshape(y_move_list, [4]))

    # 得到修正后的四个顶点坐标
    point_x = tf.add(point[:, 0], x_move)
    point_y = tf.add(point[:, 1], y_move)
    point = tf.stack([point_x, point_y], 1)
    return point


def getPixelMove(move_list):
    """
    pointLegalCheck的附加函数，返回最大的绝对值位移
    :param move_list: 四个角点与是否越界检测得到的与图片的差值 Tensor-[4]
    :return: max_move 最大位移 Tensor-[1]
    """
    move_max = tf.argmax(move_list)
    move_min = tf.argmin(move_list)
    max_move_index = tf.cond(tf.less(tf.abs(move_min), move_max),
                             true_fn=lambda: move_max,
                             false_fn=lambda: move_min
                             )
    max_move = move_list[max_move_index]
    return max_move


def mapCut(map_data, particle_map_length, old_partcile_states, new_particle_states):
    """
    给定前后坐标，根据运动方向旋转并裁剪地图，得到基于当前坐标的局部地图
    :param map_data: 原始地图数据 Tensor[height,width,channel=3] int
    :param particle_map_length: 局部地图的形状大小，默认正方形 Tensor[1] float32
    :param old_partcile_states: 前一时刻的坐标 Tensor[2](loc_x,loc_y) float32
    :param new_particle_states: 当前时刻的坐标 Tensor[2](loc_x,loc_y) float32
    :return: particle_map 基于运动方向的局部地图 Tensor[height,width,channel=3] float32
    """
    particle_map_length = particle_map_length
    map_shape = tf.shape(map_data)[0:]
    map_shape = tf.cast(map_shape, tf.float32)

    # 计算前后坐标的变化量，并以此计算运动向量的角度temp_theta
    dis = (tf.subtract(new_particle_states[1], old_partcile_states[1]),
           tf.subtract(new_particle_states[0], old_partcile_states[0]))
    temp_theta = tf.to_float(math_ops.atan2(dis[1], dis[0]))

    # 计算围绕在新坐标周围的，没有经过旋转的局部地图的四个顶点坐标
    top_left_point = [new_particle_states[0] - particle_map_length / 2,
                      new_particle_states[1] - particle_map_length / 2]
    top_right_point = [new_particle_states[0] + particle_map_length / 2,
                       new_particle_states[1] - particle_map_length / 2]
    bottom_left_point = [new_particle_states[0] - particle_map_length / 2,
                         new_particle_states[1] + particle_map_length / 2]
    bottom_right_point = [new_particle_states[0] + particle_map_length / 2,
                          new_particle_states[1] + particle_map_length / 2]

    # 计算四个顶点坐标旋转后得到的新坐标
    new_top_left_point = getRotatePoint(map_shape, new_particle_states, temp_theta, top_left_point)
    new_top_right_point = getRotatePoint(map_shape, new_particle_states, temp_theta, top_right_point)
    new_bottom_left_point = getRotatePoint(map_shape, new_particle_states, temp_theta, bottom_left_point)
    new_bottom_right_point = getRotatePoint(map_shape, new_particle_states, temp_theta, bottom_right_point)
    new_point = tf.reshape([new_top_left_point, new_top_right_point, new_bottom_left_point, new_bottom_right_point],
                           [4, 2])

    # 计算四个新坐标的横纵坐标之差的极值，该极值构成了能完整圈住旋转后的局部地图的大矩形的shape
    new_shape_x = tf.cast(new_point[tf.argmax(new_point[:, 0], 0), 0] - new_point[tf.argmin(new_point[:, 0], 0), 0],
                          tf.int32)
    new_shape_y = tf.cast(new_point[tf.argmax(new_point[:, 1], 0), 1] - new_point[tf.argmin(new_point[:, 1], 0), 1],
                          tf.int32)

    # 对四个新坐标的合法性进行判断，使其不会越出地图的边界
    new_point = pointLegalCheck(map_shape, new_point, [new_shape_x, new_shape_y])

    # 计算该大矩形的左上顶点坐标，依据大矩形的左上顶点坐标切割地图
    new_corner_x = tf.cast(new_point[tf.argmin(new_point[:, 0], 0), 0], tf.int32)
    new_corner_y = tf.cast(new_point[tf.argmin(new_point[:, 1], 0), 1], tf.int32)
    map_cut = tf.image.crop_to_bounding_box(map_data, offset_height=new_corner_y,
                                            offset_width=new_corner_x,
                                            target_height=new_shape_y,
                                            target_width=new_shape_x)

    # 反向旋转切割后的地图，使原本的运动方向与坐标轴方向（Y轴正方向）平行
    map_rotate = image.rotate(map_cut, -1.0 * temp_theta)

    # 计算位于居中的，我们最终需要的particle_map_length大小的局部地图的左上顶点，进行第二次切割
    particle_map_length = tf.to_int32(particle_map_length)
    temp_corner_width = tf.cast(tf.div(tf.subtract(new_shape_x, particle_map_length), 2), tf.int32)
    temp_corner_height = tf.cast(tf.div(tf.subtract(new_shape_y, particle_map_length), 2), tf.int32)
    particle_map = tf.image.crop_to_bounding_box(map_rotate, offset_height=temp_corner_height,
                                                 offset_width=temp_corner_width,
                                                 target_height=particle_map_length,
                                                 target_width=particle_map_length)
    return particle_map

origin_path = ".\map.jpg"
local_map_size = 130.0

def ins2npy(rewdata,ins_num):

    #get_data=get_data()
    testdata=read_data(rewdata,ins_num)
    #testdata是没有图片信息的dataframe
    #print(testdata)
    testdata["pic_data"]=None
    high=testdata.shape[0]
    high_b=high-1
    for i in range(high_b):
        print("cut pic "+str(i)+":"+str(high))
        a=testdata.loc[i,'ins_x']
        b=testdata.loc[i,'ins_y']
        old_particle_state = [a, b]
        c = testdata.loc[i+1, 'ins_x']
        d = testdata.loc[i+1, 'ins_y']
        new_particle_state = [c, d]


        '''
        保存到另外文件夹去了，改！！
        '''

        with tf.Session() as sess:
            os.chdir("D:\数据融合组\project—RL")
            file = tf.read_file(origin_path)
            map_data = tf.image.decode_image(file, channels=3)

            fig1 = plt.figure('fig1')
            plt.imshow(map_data.eval())
            map_data = tf.cast(map_data, tf.float64)

            old_particle_state = tf.convert_to_tensor(old_particle_state)
            new_particle_state = tf.convert_to_tensor(new_particle_state)
            part_map = mapCut(map_data, local_map_size, old_particle_state, new_particle_state)

            result_f = part_map.eval()
            result_f = result_f.reshape(1, -1) #local_map_size * local_map_size * 3
            np.set_printoptions(threshold=np.inf)
            testdata.set_value(i,'pic_data',result_f)

        if i == len(range(high_b))-1:
        # if i == 2:
            #testdata.to_csv( 'machine '+str(2) + '.csv', index=True)
            np.save('./insdata2npy/machine '+str(ins_num)+'.npy',testdata)
            break
            #保存为npy格式,加载使用如下
            #newdata=np.load("test123.npy") #加载后列名变为0,1,2,3.....
            #data = pd.DataFrame(newdata)

            #np.set_printoptions(threshold=np.inf)

if __name__=="__main__":
    ins2npy(rewdata,2)