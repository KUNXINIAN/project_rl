"""
输入坐标点（像素），返回该图片的向量
"""
from PIL import Image
import numpy as np
# import scipy
import matplotlib.pyplot as plt


def mapCut(pic_x,pic_y):
    #输入坐标，获取图片
    return im

def ImageToMatrix(filename):
    '''
    修改为输入im,不是filename
    '''
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
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