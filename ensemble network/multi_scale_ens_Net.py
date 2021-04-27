import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


###############################generate the average feature ############
'''print('Loading the data of VGG...')
a = np.load('ens_dataset/ens_test/test_ens_VGG1.npy')
b= np.load('ens_dataset/ens_test/test_ens_VGG2.npy')
train_VGG = np.append(a,b)
train_VGG = train_VGG.reshape(6739,55,46,64)
#a=[]
#b=[]

print('Loading the data of DenseNet...')
a = np.load('ens_dataset/ens_test/test_ens_denseNet1.npy')
b= np.load('ens_dataset/ens_test/test_ens_denseNet2.npy')
train_dense = np.append(a,b)
train_dense = train_dense.reshape(6739,55,46,64)
#a=[]
#b=[]

print('Loading the data of ResNet...')
a = np.load('ens_dataset/ens_test/test_ens_ResNet1.npy')
b= np.load('ens_dataset/ens_test/test_ens_ResNet2.npy')
train_res = np.append(a,b)
train_res = train_res.reshape(6739,55,46,64)
a=[]
b=[]

print('Loading the whole dataset...')
train_VGG = np.load('ens_train/final_ens_train_VGG.npy')
train_dense = np.load('ens_train/final_ens_train_dense.npy')
train_res = np.load('ens_train/final_ens_train_res.npy')

print('Generating the avarage train set...')
train_data = 0.33*train_VGG + 0.33*train_dense+ 0.33*train_res

np.save('ens_dataset/final_ens_test_data.npy',train_data)'''#save the average output

#train_VGG = []
#train_dense = []
#train_res = []

'''print('Loading the test data...')
D = np.load('ens_dataset/final_ens_test_data.npy')
test_data = D[3500:6739,:,:,:]
#test_data = test_data.reshape(6739,55,46,64,1)'''

'''print('Loading the train label...')
label = np.load('train_label.npy')
train_label = label[0:5000,:]
label = []'''

print('Training the classifier...')
###the presented ensemble network
model = Sequential()
model.add(Dense(4096, activation='relu', input_shape=(55,46,64)))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model_checkpoint = ModelCheckpoint('weights-multi-ens.h5', monitor='val_loss', save_best_only=True)
model.fit(train_data, train_label,epochs=50,batch_size=12,shuffle=True, validation_split=0.01,callbacks=[model_checkpoint])

'''print('Loading saved weights...')
model.load_weights('weights-multi-ens.h5')

print('Predicting sorts on test data...')
imgs_predict_test = model.predict(test_data, batch_size=12, verbose=1)
np.save('ens_dataset/multiscale_ens_predict2.npy', imgs_predict_test)'''
