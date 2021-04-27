import numpy as np

predict=np.load('ens_ave_results.npy')
label=np.load('test_label.npy')

results=[]

for i in range(6739):
    if predict[i][0] > predict[i][1]:
        results= np.append(results,[1.,0.])
    else:
        results=np.append(results,[0.,1.])
results=results.reshape(6739,2)

num=0

for j in range(6739):
    if results[j][0] == label[j][0]:
        num=num+1

accuracy = num/6739

print(accuracy)
