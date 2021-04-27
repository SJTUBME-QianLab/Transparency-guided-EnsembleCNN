# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:52:25 2019

@author: xiaoming liu
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics  ###计算roc和auc

y_test = np.load('test_label.npy')
label = np.zeros(shape=(77,))

for i in range(77):
    if y_test[i][0] > y_test[i][1]:
        label[i] = 1
    else:
        label[i] = 0
        
#label = np.array([1,1,0,1,1,1,1,1,1,0,0,1,0,1,0,0,0,0,0,1,0,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,1,1,0])
    
y_s = np.load('multiscale_ens_predict.npy')
y_score=y_s[:77]
#y_score = np.array([[0.551698,0.448302],[0.606885,0.393115],[0.276452,0.723548],[0.700721,0.299279],[0.60684,0.39316],[0.538423,0.461577],[0.749755,0.250245],[0.486141,0.513859],[0.816907,0.183093],[0.349445,0.650555],[0.442831,0.557169],[0.54267,0.45733],[0.617624,0.382376],[0.333772,0.666228],[0.4862,0.5138],[0.502794,0.497206],[0.732991,0.267009],[0.292794,0.707206],[0.45919,0.54081],[0.630158,0.369842],[0.443157,0.556843],[0.453416,0.546584],[0.682329,0.317671],[0.68097,0.31903],[0.764123,0.235877],[0.443377,0.556623],[0.59124,0.40876],[0.512353,0.487647],[0.627828,0.372172],[0.336533,0.663467],[0.516729,0.483271],[0.575293,0.424707],[0.496598,0.503402],[0.423016,0.576984],[0.624157,0.375843],[0.251894,0.748106],[0.485828,0.514172],[0.542601,0.457399],[0.575307,0.424693],[0.461612,0.538388]])

score = y_score[:,0]


precision, recall, thresholds = metrics.precision_recall_curve(label,score)
auc = metrics.auc(recall, precision)
lw = 2
plt.figure(figsize=(5,5))
plt.plot(precision, recall, color='red', lw=lw)
plt.plot([0, 1], [1, 0], color='gray', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

font = {'family' : 'Times New Roman','weight' : 'normal','size' : 22}
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
plt.xlabel('Recall',font)
plt.ylabel('Precision',font)#
#plt.title('Ensemble Network',font)
plt.title('P-R',font)
#plt.title('Ensemble Networks with best features',font)
#plt.text(0.09,1.13,'Ensemble Network',font)
#plt.text(0.05,1.03,'best and suboptimal features',font)
plt.show()

