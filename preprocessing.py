# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import lfilter, hamming
#from pylab import plot

#帧的类
class frame:
    """audio:一帧的原数据  window:窗  start:这一帧的起始取样点"""
    def __init__(self, audio, window, start):
        self.start = start
        self.data = window * audio #因为下标个数一致，直接对应相乘加窗
        self.energy = sum(self.data**2) #计算短时能量 Reference:http://www.docin.com/p-1055762767.html


def preprocessing(music, fs, frameTime):
    """预处理实现：*统一精度、分帧、加窗、筛除静音帧"""
    #对目前的程序，音频使用何种采样精度不会影响提取到的mfcc值
    #如果添加了其他对精度敏感的操作，对于int16与int32格式的数据，可用以下代码将其归一化为float(0,1)
    #if music.dtype == 'int16':
    #    music = music/(2**15)
    #elif music.dtype == 'int32':
    #    music = music/(2**31)
    #或是统一为16bit精度
    #if music.dtype == 'float32':
    #    music = music*(2**15)
    #elif music.dtype == 'int32':
    #    music = music/(2**16)

    #预加重
    b = np.array([1, -0.97])
    left = lfilter(b, 1, music[:,0])
    right = lfilter(b, 1, music[:,1])
    #left,right:预加重后的左右声道
    #plot(left)
    #plot(right)

    #分帧加窗
    #Reference:
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hamming.html
    frameLength = int(fs * frameTime) #以取样点数表示的帧长
    #即是 frameLength = int(fs/(1/frameTime))
    frameShift = int(frameLength/2) #帧移
    hammingWindow = hamming(frameLength) #生成汉明窗

    f = [] #记录所有帧对象的列表
    for i in range(0, len(left) - frameLength, frameShift): #音频最后长度不足一帧的部分需舍去
        f.append(frame(left[i:i+frameLength], hammingWindow, i))
    for i in range(0, len(right) - frameLength, frameShift): #对左右声道独立地分帧，在列表中右声道接在左声道后面
        f.append(frame(right[i:i+frameLength], hammingWindow, i))

    #计算所有帧的短时能量，算出阈值
    energy = np.array([frame.energy for frame in f]) #短时能量
    Lenergy = np.amin(energy) + (np.amax(energy) - np.amin(energy))*0.03 #能量阈值：把与最低能量相距3%以内的帧认为是静音帧
    f = [frame for frame in f if frame.energy > Lenergy] #更新对象列表，只保留非静音帧

    return f, frameLength
