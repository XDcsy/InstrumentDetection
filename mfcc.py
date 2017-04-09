# -*- coding: utf-8 -*-
import numpy as np

from scipy.io import wavfile
from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct
#from pylab import plot

#帧的类
class frame:
    """audio:一帧的原数据  window:窗  start:这一帧的起始取样点"""
    def __init__(self, audio, window, start):
        self.start = start
        self.data = window * audio #因为下标个数一致，直接对应相乘加窗
        self.energy = sum(self.data**2) #计算短时能量 Reference:http://www.docin.com/p-1055762767.html

###############################################################
# Source code for the following part come from: scikit.talkbox
# file: segmentaxis.py
# Author: cournape
# https://github.com/cournape/talkbox
# Is edited to suit this program

def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    # Compute the filter bank
    # Compute start/middle/end points of the triangular filters in spectral domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])
    return fbank

def mfcc(f, fs, frameLength, nceps=13):
    nfft = frameLength * 2
    lowfreq = 133.33
    #highfreq = 6855.4976
    linsc = 200/3.
    logsc = 1.0711703
    #三角滤波器组的几个参数

    nlinfil = 13
    nlogfil = 27
    #滤波器的个数

    fbank = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)
    data = np.array([frame.data for frame in f]) #所有帧的内容
    # Compute the spectrum magnitude
    spec = np.abs(fft(data, nfft, axis=-1))
    # Filter the spectrum through the triangle filterbank
    mspec = np.log10(np.dot(spec, fbank.T))
    #由于通过短时能量筛选去除了静音帧，理论上此处不会出现系数为0的情况
    #如果删除了排除静音帧的步骤，有可能会存在0系数导致无法计算，此时可用下方代码替代
    #epsilon = 1e-6
    #mspec = np.log10(np.dot(np.maximum(spec, epsilon), fbank.T))
    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, 1:nceps]
    # 一般取DCT后的第2个到第13个系数作为MFCC系数
    return ceps

# End of the codes from scikit.talkbox
###############################################################


#读取音频
#Reference:
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#scipy.io.wavfile.read
#fs:采样率，录音设备在一秒钟内对声音信号的采样次数
#music:原始音频
fs, music = wavfile.read(r'e:\shutdown.wav')
if music.dtype.type not in [np.int16, np.int32, np.float32]:
    raise TypeError('only 16bit,32bit PCM and 32bit floating-point wavefiles are supported')
#只支持这三种精度的wav文件，注意不支持8bit与24bit
#对此程序，音频使用何种采样精度不会影响提取到的mfcc值
#如果添加了其他对精度敏感的操作，对于int16与int32格式的数据，可用以下代码将其归一化
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
b = np.array([1, -0.95])
left = lfilter(b, 1, music[:,0])
right = lfilter(b, 1, music[:,1])
#left,right:预加重后的左右声道
#plot(left)
#plot(right)

#分帧加窗
#Reference:
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hamming.html
frameTime = 0.02 #(s)
frameLength = int(fs/(1/frameTime)) #以取样点数表示的帧长
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

#对各帧计算出mfcc
ceps = mfcc(f, fs, frameLength)