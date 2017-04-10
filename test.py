import os
import re
import pickle
import numpy as np

from scipy.io import wavfile
from preprocessing import preprocessing
from mfcc import mfcc

#读取模型
clf = pickle.load(open('model_svm', 'rb'))
#读取标签对应的乐器名称
instruments = pickle.load(open('names', 'rb'))

#读取测试文件目录 
musicpath = input("Enter the path of the folder containing testing audios :")
print("Computing...")
filename = [name for name in os.listdir(musicpath) if re.match(r'.*wav',name)]
result = []
for i in range(len(filename)):
    fs, music = wavfile.read(os.path.join(musicpath, filename[i]))
    if music.dtype.type not in [np.int16, np.int32, np.float32]:
        raise TypeError('only 16bit,32bit PCM and 32bit floating-point wavefiles are supported')
        #只支持这三种精度的wav文件，注意不支持8bit与24bit

    #预处理
    frameTime = 0.02 #(s)
    #f：处理后的各帧对象 frameLength：以取样点数表示的帧长
    f, frameLength = preprocessing(music, fs, frameTime)

    #计算出mfcc
    ceps = mfcc(f, fs, frameLength)
    result.append(clf.predict(ceps)) #得到对每一帧的预测结果

for i in range(len(result)):
    result[i] = np.argmax(np.bincount(result[i])) #每帧投票，选出得票最多的结果
    print('The instrument played in "'+filename[i]+'" is detected as: '+instruments[result[i]])
