# -*- coding: utf-8 -*-
import os
import re
import numpy as np

from scipy.io import wavfile
from preprocessing import preprocessing
from mfcc import mfcc

def generate(dirname, instrumentPath):
    """对每个乐器文件夹内的音频，生成mfcc"""
    #读取当前目录下音频的文件名
    filename = [name for name in os.listdir(instrumentPath) if re.match(r'.*wav',name)]
    for i in range(len(filename)):
        #fs:采样率，录音设备在一秒钟内对声音信号的采样次数
        #music:原始音频
        fs, music = wavfile.read(os.path.join(os.path.join(root, dirname), filename[i]))
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

    np.save(dirname, ceps) #保存mfcc参数


print("*Make sure the training data is arranged in the structure given in README.")
musicpath = input("Enter the path of the folder containing training audios :")
for root, dirnames, filenames in os.walk(musicpath): #分别为：父目录、文件夹名、文件名
    for dirname in dirnames: #遍历所有子文件夹
        print("Found instument: "+dirname+". Generating it's MFCC...")
        instrumentPath = os.path.join(root, dirname)
        generate(dirname, instrumentPath) #对每个子文件夹，生成mfcc
