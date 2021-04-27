import cv2    
import numpy as np

imgs_train = np.load('multi_all_data.npy')
print('Data has been read!')

cols=218
rows=182
###rotation in different scale
trans_rot1=[];
for i in range(13480):
    img = imgs_train[i]
    M1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 1, 1) 
    T1= cv2.warpAffine(img, M1, (cols, rows))
    trans_rot1=np.append(trans_rot1,T1)
print('1 time finished!')

trans_rot2=[];
for j in range(13480):
    img = imgs_train[j]
    M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 2, 1)
    T2= cv2.warpAffine(img, M2, (cols, rows))
    trans_rot2=np.append(trans_rot2,T2)
print('2 times finished!')

trans_rot3=[];
for k in range(13480):
    img = imgs_train[k]
    M3 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 3, 1)
    T3= cv2.warpAffine(img, M3, (cols, rows))
    trans_rot3=np.append(trans_rot3,T3)
print('3 times finished!')

trans_rot4=[];
for m in range(13480):
    img = imgs_train[m]
    M4 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 4, 1)
    T4= cv2.warpAffine(img, M4, (cols, rows))
    trans_rot4=np.append(trans_rot4,T4)
print('4 times finished!')

#trans_rot5=[];
#for n in range(1348):
    #img = imgs_train[n]
    #M5 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    #T5= cv2.warpAffine(img, M5, (cols, rows))
    #trans_rot5=np.append(trans_rot5,T5)
#print('5 times finished!')

trans_r1=np.append(trans_rot1,trans_rot2)
trans_r2=np.append(trans_r1,trans_rot3)
#trans_r3=np.append(trans_r2,trans_rot4)
trans_rot = np.append(trans_r2,trans_rot4)

print('Saving results.....')
trans_rot = trans_rot.reshape(13480*4,cols,rows)
np.save('all-rot2.npy',trans_rot)
print('Done!')

