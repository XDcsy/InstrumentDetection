# -*- coding: utf-8 -*-
from scipy import signal
from scipy.io import wavfile
import numpy as np
#from pylab import plot

#帧的类
#audio:一帧的原数据
#window:窗
#start:这一帧的起始取样点
class frame:
	def __init__(self, audio, window, start):
		self.start = start
		self.data = window * audio #因为下标个数一致，直接对应相乘加窗
		self.energy = sum(self.data**2) #计算短时能量 Reference:http://www.docin.com/p-1055762767.html



#读取音频
#Reference:
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#scipy.io.wavfile.read
#http://www.mamicode.com/info-detail-1180317.html
#freq:采样率，录音设备在一秒钟内对声音信号的采样次数
#music:原始音频
freq, music = wavfile.read(r'e:\shutdown.wav')
if music.dtype.type not in [np.int16, np.int32, np.float32]:
    raise TypeError('only 16bit,32bit PCM and 32bit floating-point wavefiles are supported')
#只支持这三种精度的wav文件，注意不支持8bit与24bit
#对于int16与int32格式的数据，将其归一化
if music.dtype == 'int16':
    music = music/(2**15)
elif music.dtype == 'int32':
    music = music/(2**31)

#预加重
#Reference:
#http://old.sebug.net/paper/books/scipydoc/filters.html
#http://blog.sina.com.cn/s/blog_e9d8e2610102uxgv.html
#http://www.docin.com/p-318611828.html
#http://www.mamicode.com/info-detail-502860.html
b = np.array([1, -0.95])
left = signal.lfilter(b, 1, music[:,0])
right = signal.lfilter(b, 1, music[:,1])
#left,right:预加重后的左右声道
#plot(left)
#plot(right)

#分帧加窗
#Reference:
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hamming.html
frameTime = 0.02 #(s)
frameLength = int(freq/(1/frameTime)) #以取样点数表示的帧长
frameShift = int(frameLength/2) #帧移
hammingWindow = signal.hamming(frameLength) #生成汉明窗

f = [] #记录所有帧对象的列表
for i in range(0, len(left) - frameLength, frameShift): #音频最后长度不足一帧的部分需舍去
	f.append(frame(left[i:i+frameLength], hammingWindow, i))
for i in range(0, len(right) - frameLength, frameShift): #对左右声道独立地分帧，在列表中右声道接在左声道后面
	f.append(frame(right[i:i+frameLength], hammingWindow, i))

#计算所有帧的短时能量，算出阈值
energy = np.array([frame.energy for frame in f]) #短时能量
Lenergy = np.amin(energy) + (np.amax(energy) - np.amin(energy))*0.03 #能量阈值：把与最低能量相距3%以内的帧认为是静音帧
f = [frame for frame in f if frame.energy > Lenergy] #更新对象列表，只保留非静音帧
