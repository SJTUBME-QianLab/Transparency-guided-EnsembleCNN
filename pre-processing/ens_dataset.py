import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
from keras.models import Model
from keras import models
from keras.layers import Input, Dense,MaxPooling2D, BatchNormalization, Concatenate, Dropout, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Add, Flatten, GlobalMaxPooling2D, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

##################################################################################
#generate the average dataset to train the ensemble network

print('Loading the first net...')
#VGG16
TRAIN_CLASSIFY_USE_BN = True

def get_simplified_VGG_classifier():
    inputs = Input((218,182,1))

    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

    x = GlobalMaxPooling2D()(x)


    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

'''print('Loading the second net...')

# ResNet
RESNET_BLOCKS = 6
RESNET_SHRINKAGE_STEPS = 5
RESNET_INITIAL_FILTERS = 4

def conv_bn_relu(x, filters, kernel_size=(5, 5), strides=(1, 1), padding='same', apply_relu=True):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if apply_relu:
        x = Activation('relu')(x)
    return x

def bottleneck(x, shrinkage=False):
    #print('resnet block, shrinkage:{}'.format(shrinkage))
    #print(x.get_shape())

    input_filters = x.get_shape()[3].value
    keep_filters = input_filters // 2 if shrinkage else input_filters // 4
    output_filters = input_filters * 2 if shrinkage else input_filters
    first_strides = (2, 2) if shrinkage else (1, 1)

    residual = conv_bn_relu(x, filters=keep_filters, kernel_size=(5, 5), strides=first_strides)
    residual = conv_bn_relu(residual, filters=keep_filters, kernel_size=(5, 5))
    residual = conv_bn_relu(residual, filters=output_filters, kernel_size=(5, 5), apply_relu=True)

    if shrinkage:
        x = conv_bn_relu(x, filters=output_filters, kernel_size=(5, 5), strides=(2, 2), apply_relu=True)

    print(residual.get_shape())
    print(x.get_shape())
    x = Add()([residual, x])
    x = Activation('relu')(x)

    return x

def get_ResNet_classifier():
    inputs = Input((218, 182, 1))

    x = conv_bn_relu(inputs, RESNET_INITIAL_FILTERS)

    print('base')
    print(x.get_shape())

    for i in range(RESNET_BLOCKS):
        x = bottleneck(x, shrinkage=(i % RESNET_SHRINKAGE_STEPS != 0))

    print('top')
    x = GlobalMaxPooling2D()(x)
    print(x.get_shape())

    x = Dense(2, activation='softmax')(x)
    print(x.get_shape())

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

    return model

print('Loading the third net...')
#DenseNet
DENSE_NET_BLOCKS = 3
DENSE_NET_BLOCK_LAYERS = 5
DENSE_NET_INITIAL_CONV_DIM = 8
DENSE_NET_GROWTH_RATE = DENSE_NET_INITIAL_CONV_DIM // 2
# called DenseNet-BC if ENABLE_BOTTLENETCK and COMPRESSION < 1 in paper
DENSE_NET_ENABLE_BOTTLENETCK = False
DENSE_NET_TRANSITION_COMPRESSION = 1.0
DENSE_NET_ENABLE_DROPOUT = True
DENSE_NET_DROPOUT = 0.5

def bn_relu_conv(x, filters, kernel_size=(3, 3)):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)

    return x

def dense_block(x):
    print('dense block')
    print(x.get_shape())

    for _ in range(DENSE_NET_BLOCK_LAYERS):
        y = x

        if DENSE_NET_ENABLE_BOTTLENETCK:
            y = bn_relu_conv(y, filters=DENSE_NET_GROWTH_RATE, kernel_size=(1, 1))

        y = bn_relu_conv(y, filters=DENSE_NET_GROWTH_RATE, kernel_size=(3, 3))
        x = Concatenate(axis=3)([x, y])
        print(x.get_shape())

    return x

def transition_block(x):
    print('transition block')
    print(x.get_shape())

    filters = x.get_shape()[3].value
    filters = int(filters * DENSE_NET_TRANSITION_COMPRESSION)

    x = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(x)
    x = AveragePooling2D(pool_size=(2, 2), padding='same')(x)
    print(x.get_shape())

    return x

def get_DenseNet_classifier():
    inputs = Input((218,182,1))
    x = Conv2D(DENSE_NET_INITIAL_CONV_DIM, (3, 3), padding='same')(inputs)
    print('input')
    print(x.get_shape())

    for i in range(DENSE_NET_BLOCKS):
        x = dense_block(x)
        if i != DENSE_NET_BLOCKS - 1:
            x = transition_block(x)

    print('top')
    x = GlobalAveragePooling2D()(x)
    print(x.get_shape())

    if DENSE_NET_ENABLE_DROPOUT:
        x = Dropout(DENSE_NET_DROPOUT)(x)

    x = Dense(2, activation='softmax')(x)
    print(x.get_shape())

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])

    return model'''

#############################################################################################
print('Loading the train data...')
imgs_train = np.load('compare/train_data_small.npy')
mean = np.mean(imgs_train)
std = np.mean(imgs_train)
    
imgs_train -= mean
imgs_train /= std

train_datasets = np.zeros(shape=(5000,55,46,64))
temp = np.zeros(shape=(55,46,64))

model1= get_simplified_VGG_classifier()
#model1.summary()
model1.load_weights('VGG_weights.h5')

#model2= get_ResNet_classifier()
#model2.summary()
#model2.load_weights('ResNet_weights2.h5')

#model3= get_DenseNet_classifier()
#model3.load_weights('DenseNet_weights.h5')

print('Generating the train set...')
for i in range(5000):
    train = imgs_train[i].reshape(1,218,182,1)
    
    preds1 = model1.predict(train, batch_size=1, verbose=1)
    a=np.argmax(preds1[0])
    output1 = model1.output[:, a]
    conv_name_layer1 = model1.get_layer('conv2d_5')
    grads1 = K.gradients(output1, conv_name_layer1.output)[0]
    pooled_grads1 = K.mean(grads1, axis=(0, 1, 2))
    iterate1 = K.function([model1.input], [pooled_grads1, conv_name_layer1.output[0]])
    pooled_grads_value, output_value1 = iterate1([train])
    '''
    preds2 = model2.predict(train, batch_size=1, verbose=1)
    b=np.argmax(preds2[0])
    output2 = model2.output[:, b]
    conv_name_layer2 = model2.get_layer('activation_25')
    grads2 = K.gradients(output2, conv_name_layer2.output)[0]
    pooled_grads2 = K.mean(grads2, axis=(0, 1, 2))
    iterate2 = K.function([model2.input], [pooled_grads2, conv_name_layer2.output[0]])
    pooled_grads_value, output_value2 = iterate2([train])

    preds3 = model3.predict(train, batch_size=1, verbose=1)
    c=np.argmax(preds3[0])
    output3 = model3.output[:, c]
    conv_name_layer3 = model3.get_layer('activation_15')
    grads3 = K.gradients(output3, conv_name_layer3.output)[0]
    pooled_grads3 = K.mean(grads3, axis=(0, 1, 2))
    iterate3 = K.function([model3.input], [pooled_grads3, conv_name_layer3.output[0]])
    pooled_grads_value, output_value3 = iterate3([train])'''
    for j in range(64):
        img1 = cv2.resize(output_value1[:,:,j],(46,55))
        #img2 = cv2.resize(output_value2[:,:,j],(46,55))
        #img = 0.33*img1 + 0.33*img2 + 0.33*output_value3[:,:,j]
        temp[:,:,j] = img1

    train_datasets[i] = temp
    print(' %d image is finished!' %(i))

np.save('compare/2nd_test_ens_ResNet.npy',train_datasets)
print('Train set has been generated!')
#################################################################################
'''print('Loading the test data...')
imgs_test = np.load('test_data.npy')

#imgs_test -= mean
#imgs_test /= std

test_datasets = np.zeros(shape=(6739,55,46,64))
temp2 = np.zeros(shape=(55,46,64))

print('Generating the test set...')
for m in range(6739):
    test = imgs_test[m].reshape(1,218,182,1)
    preds1 = model1.predict(test[m], batch_size=1, verbose=1)
    a=np.argmax(preds1[0])
    output1 = model1.output[:, a]
    conv_name_layer1 = model1.get_layer('conv2d_5')
    grads1 = K.gradients(output1, conv_name_layer1.output)[0]
    pooled_grads1 = K.mean(grads1, axis=(0, 1, 2))
    iterate1 = K.function([model1.input], [pooled_grads1, conv_name_layer1[0]])
    pooled_grads_value, output_value1 = iterate1([test])

    preds2 = model2.predict(imgs_test[m], batch_size=1, verbose=1)
    b=np.argmax(preds2[0])
    output2 = model2.output[:, b]
    conv_name_layer2 = model2.get_layer('activation_25')
    grads2 = K.gradients(output2, conv_name_layer2.output)[0]
    pooled_grads2 = K.mean(grads2, axis=(0, 1, 2))
    iterate2 = K.function([model2.input], [pooled_grads2, conv_name_layer2[0]])
    pooled_grads_value, output_value2 = iterate2([imgs_test[m]])

    preds3 = model3.predict(imgs_test[m], batch_size=1, verbose=1)
    c=np.argmax(preds3[0])
    output3 = model3.output[:, c]
    conv_name_layer3 = model3.get_layer('activation_15')
    grads3 = K.gradients(output3, conv_name_layer3.output)[0]
    pooled_grads3 = K.mean(grads3, axis=(0, 1, 2))
    iterate3 = K.function([model3.input], [pooled_grads3, conv_name_layer3[0]])
    pooled_grads_value, output_value3 = iterate3([imgs_test[m]])
    for n in range(64):
        img1 = cv2.resize(output_value1[:,:,n],(46,55))
        img2 = cv2.resize(output_value2[:,:,n],(46,55))
        img = 0.33*img1 + 0.33*img2 + 0.33*output_value3[:,:,j]
        tempn[:,:,j] = img

    train_datasets[m] = temp2
    print(' %d image is finished!' %(m+1))
print('Test set has been generated!')


print('ALL DONE!!!')'''
        
                             
        

    
