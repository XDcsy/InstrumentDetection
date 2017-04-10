# -*- coding: utf-8 -*-
import os
import re
import pickle
from sklearn import svm
import numpy as np

insrumentName = [name for name in os.listdir(os.getcwd()) if re.match(r'.*npy',name)]
ceps = []
for i in range(len(insrumentName)):
    ceps.append(np.load(insrumentName[i])) #读取mfcc参数

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(ceps, insrumentName)
pickle.dump(clf,open('model_svm','wb'))
# getback = pickle.load(open('model_svm', 'rb'))
