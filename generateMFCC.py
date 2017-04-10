# -*- coding: utf-8 -*-
import os
import re
import numpy as np

from scipy.io import wavfile
from preprocessing import preprocessing
from mfcc import mfcc
#获取此文件所在文件夹的名称，用于在保存结果时命名
foldername = os.path.basename(os.getcwd())

#读取当前目录下音频的文件名
filename = [name for name in os.listdir(os.getcwd()) if re.match(r'.*wav',name)]
for i in range(len(filename)):
    #fs:采样率，录音设备在一秒钟内对声音信号的采样次数
    #music:原始音频
    fs, music = wavfile.read(filename[i])
    if music.dtype.type not in [np.int16, np.int32, np.float32]:
        raise TypeError('only 16bit,32bit PCM and 32bit floating-point wavefiles are supported')
        #只支持这三种精度的wav文件，注意不支持8bit与24bit

    #预处理
    frameTime = 0.02 #(s)
    #f：处理后的各帧对象 frameLength：以取样点数表示的帧长
    f, frameLength = preprocessing(music, fs, frameTime)

    #计算出mfcc
    tmpceps = mfcc(f, fs, frameLength)
    if (i == 0):
        ceps = tmpceps
    else:
        ceps = np.concatenate((ceps, tmpceps), axis=0)

np.save(foldername, ceps) #保存mfcc参数
#ceps = np.load("name") #读取mfcc参数
