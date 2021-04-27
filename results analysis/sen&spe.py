import numpy as np

label=np.load('test_label.npy')
prediction=np.load('ens_ave_results.npy')

results=np.zeros(shape=(6739,2))

for i in range(6739):
    if prediction[i][0] > prediction[i][1]:
        results[i] = [1.,0.]
    else:
        results[i] = [0.,1.]

tp = 0;  
fn = 0;  
fp = 0;  
tn = 0;

for i in range(6739):
    if results[i][0] ==1 and label[i][0] ==1:
        tp = tp+1
    elif results[i][0] ==1 and label[i][0] ==0:
        fp=fp+1
    elif results[i][0] ==0 and label[i][0] ==1:
        fn=fn+1
    elif results[i][0] ==0 and label[i][0] ==0:
        tn=tn+1


sens = tp/(tp+fn) 
spec = tn/(tn+fp)  
pre = tp/(tp+fp)  
rec = sens  
print ("The sensitivity is %f" %(sens)),
print ("The specificity is %f"%(spec)),
print ("The precision is %f"%(pre)),
print ("The recall is %f"%(rec)),

