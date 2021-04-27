
import SimpleITK as sitk
import xlrd
from keras.utils import np_utils
import numpy as np
#######generate the experiment dataset#####

info =xlrd.open_workbook('data/label.xlsx')
table = info.sheets()[0]
nrows = table.nrows

print(nrows)

print('Excel has been import')

# ************* step 1: obtaining the shape information  ****************
total_slices = 0
for i in range(nrows):
    rowContent = table.row_values(i)
    ID = int(rowContent[0])
    if i==1:
        d = sitk.ReadImage('data/' + str(ID) + '.nii')
        img_array = sitk.GetArrayFromImage(d)
        [thickness,height,width]=img_array.shape

    if (rowContent[1]==''):
        print('The %d patient is skipped!' %(i+1))
        continue
    else:
        start_slice=int(rowContent[1])
        end_slice=int(rowContent[2])
        num_slice = end_slice-start_slice+1
        total_slices = total_slices+num_slice

print("Total slices number is %d :", total_slices)

# ************* step 2: importing the valuable slices as 3D array and label vector ****************

img_datasets = np.zeros(shape=(total_slices,height,width))
labels = np.zeros(total_slices)
k=0
for i in range(nrows):
    rowContent = table.row_values(i)
    ID = int(rowContent[0])

    if (rowContent[1]==''):
        print('The %d patient is skipped!' %(i+1))
        continue
    else:
        start_slice=int(rowContent[1])
        end_slice=int(rowContent[2])
        label = int(rowContent[3])
        num_slice = end_slice-start_slice+1

        d = sitk.ReadImage('data/' + str(ID) + '.nii')
        img_array = sitk.GetArrayFromImage(d)
        [thickness,height,width]=img_array.shape
        for j in range((start_slice-1),end_slice):
            img=img_array[j]
            img_datasets[k] = img
            labels[k] = label
            k+=1

print("MRI of %d patients have been read!" % (i+1))


# ************* step 3: One Hot Encoding and save ****************

label_OneHot = np_utils.to_categorical(labels)

np.save('C:/Users/22932/Desktop/1/data_Qxh.npy',img_datasets)
np.save('C:/Users/22932/Desktop/1/label_Qxh.npy',label_OneHot)
print('Data and label have been saved!')









