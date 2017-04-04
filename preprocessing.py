# -*- coding: utf-8 -*-
from scipy.io import wavfile
from numpy import array
from scipy import signal
#from pylab import plot

#帧的类
#audio:一帧的原数据
#window:窗
#start:这一帧的起始取样点
class frame:
    def __init__(self, audio, window, start):
        self.start = start
        self.data = window * audio #因为下标个数一致，直接对应相乘加窗


#读取音频
#Reference:
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#scipy.io.wavfile.read
#freq:采样率，录音设备在一秒钟内对声音信号的采样次数
#music:原始音频
freq, music = wavfile.read(r'e:\shutdown.wav')

#预加重
#Reference:
#http://old.sebug.net/paper/books/scipydoc/filters.html
#http://blog.sina.com.cn/s/blog_e9d8e2610102uxgv.html
#http://www.docin.com/p-318611828.html
#http://www.mamicode.com/info-detail-502860.html
b = array([1, -0.95])
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

f = []
for i in range(0, len(left) - frameLength, frameShift): #最后长度不足一帧的部分需舍去
    f.append(frame(left[i:i+frameLength], hammingWindow, i))
for i in range(0, len(right) - frameLength, frameShift): #对左右声道独立地做分帧
    f.append(frame(right[i:i+frameLength], hammingWindow, i))




