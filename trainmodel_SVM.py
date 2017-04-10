# -*- coding: utf-8 -*-
import os
import re
import pickle
from sklearn import svm
import numpy as np

insrumentName = [name for name in os.listdir(os.getcwd()) if re.match(r'.*npy',name)]
lable = []
for i in range(len(insrumentName)):
    if (i == 0):
        ceps = np.load(insrumentName[i])  #读取mfcc参数
        lable.extend([i]*ceps.shape[0])  #生成标签
    else:
        tmpceps = np.load(insrumentName[i])
        ceps = np.concatenate((ceps, tmpceps), axis=0)
        lable.extend([i]*tmpceps.shape[0])
        
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(ceps, lable)
pickle.dump(clf,open('model_svm','wb'))
pickle.dump(insrumentName,open('names','wb'))
# getback = pickle.load(open('model_svm', 'rb'))
