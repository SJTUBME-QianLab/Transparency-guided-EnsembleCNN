import cv2    
import numpy as np

print('Data has been read!')
imgs_train = np.load('data.npy')
cols=218
rows=182
##translation in different scales
trans_pingyi1=[];
for i in range(1348):
    img = imgs_train[i]
    M1 = np.float32([[1, 0, 24], [0, 1, 20]])
    T1= cv2.warpAffine(img, M1, (cols, rows))
    trans_pingyi1=np.append(trans_pingyi1,T1)
print('1 time finished!')

trans_pingyi2=[];
for j in range(1348):
    img = imgs_train[j]
    M2 = np.float32([[1, 0, 12], [0, 1, 10]])
    T2= cv2.warpAffine(img, M2, (cols, rows))
    trans_pingyi2=np.append(trans_pingyi2,T2)
print('2 times finished!')

trans_pingyi3=[];
for k in range(1348):
    img = imgs_train[k]
    M3 = np.float32([[1, 0, 6], [0, 1, 5]])
    T3= cv2.warpAffine(img, M3, (cols, rows))
    trans_pingyi3=np.append(trans_pingyi3,T3)
print('3 times finished!')

trans_pingyi4=[];
for m in range(1348):
    img = imgs_train[m]
    M4 = np.float32([[1, 0, 30], [0, 1, 25]])
    T4= cv2.warpAffine(img, M4, (cols, rows))
    trans_pingyi4=np.append(trans_pingyi4,T4)
print('4 times finished!')

trans_p1 = np.append(trans_pingyi1,trans_pingyi2)
trans_p2 = np.append(trans_p1,trans_pingyi3)
trans_pingyi = np.append(trans_p2,trans_pingyi4)

print('Saving results.....')
trans_pingyi = trans_pingyi.reshape(1348*4,cols,rows)
np.save('all-pingyi.npy',trans_pingyi)
print('Done!')

