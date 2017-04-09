# -*- coding: utf-8 -*-
import numpy as np

from scipy.io import wavfile
from preprocessing import preprocessing
from mfcc import mfcc

#读取音频
#fs:采样率，录音设备在一秒钟内对声音信号的采样次数
#music:原始音频
fs, music = wavfile.read(r'e:\shutdown.wav')
if music.dtype.type not in [np.int16, np.int32, np.float32]:
    raise TypeError('only 16bit,32bit PCM and 32bit floating-point wavefiles are supported')
#只支持这三种精度的wav文件，注意不支持8bit与24bit

#预处理
frameTime = 0.02 #(s)
#f：处理后的各帧对象 frameLength：以取样点数表示的帧长
f, frameLength = preprocessing(music, fs, frameTime)
#计算出mfcc
ceps = mfcc(f, fs, frameLength)
