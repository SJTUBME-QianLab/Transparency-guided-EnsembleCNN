import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics  ###get roc curves and auc

y_test = np.load('test_label.npy')
label = np.zeros(shape=(6739,))

for i in range(6739):
    if y_test[i][0] > y_test[i][1]:
        label[i] = 1
    else:
        label[i] = 0
    
y_score = np.load('VGG_prediction.npy')
score = y_score[:,0]
#score = score.reshape(6739,1)


# Compute ROC curve and ROC area for each class
fpr,tpr,threshold = metrics.roc_curve(label, score) 
roc_auc = metrics.auc(fpr,tpr) ###get auc
 
plt.figure()
lw = 2
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, color='darkorange',lw=lw) 
plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
plt.text(0.42,0.05,'ROC curve (area = %0.2f)' % roc_auc,size = 13)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

font = {'family' : 'Times New Roman','weight' : 'normal','size'   : 20}
plt.xlabel('False Positive Rate',font)
plt.ylabel('True Positive Rate',font)
plt.title('DenseNet',font)
plt.legend(loc="lower right")
plt.show()
