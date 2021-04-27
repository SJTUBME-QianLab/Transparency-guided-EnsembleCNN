from keras.models import Model
from keras.layers import Input, Conv2D, Dense, BatchNormalization, Add, Flatten, Concatenate, AveragePooling2D, GlobalMaxPooling2D, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import time
import numpy as np

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

def classify_train():
    print('start classify_train')


    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    imgs_train = np.load('train_data.npy')#加载训练集数据
    labels_train = np.load('train_label.npy')
    ##imgs_train = imgs_train.reshape(11940,218,182,1) 

    print('-'*30)
    print('Shuffling the data...')
    print('-'*30)

    #index=[i for i in range(11940*5)]
    #random.shuffle(index)
    #imgs_train=imgs_train[index]
    #labels_train=labels_train[index]

    
    mean = np.mean(imgs_train)
    std = np.mean(imgs_train)
    
    imgs_train -= mean
    imgs_train /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model= get_ResNet_classifier()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    ##tb_cb = TensorBoard(log_dir=log_filepath, write_images=False, histogram_freq=1, write_graph=True)

    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(imgs_train, labels_train, batch_size=40, epochs=3, verbose=1, shuffle=False, validation_split=0.01,callbacks=[model_checkpoint])  
    
    print('-'*30)
    print('Loading test data...')
    print('-'*30)
    imgs_test = np.load('test_data.npy')#加载测试集数据
    
    #imgs_test = imgs_test.reshape(154,218,182,1)
    
    imgs_test -= mean
    imgs_test /= std 

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting sorts on test data...')
    print('-'*30)
    imgs_predict_test = model.predict(imgs_test, batch_size=40, verbose=1)
    np.save('imgs_predict_test.npy', imgs_predict_test)

    


if __name__ == '__main__':
    classify_train()


    

#endtime = time.clock()
#print("The train_and_predict running time is %g s" %(endtime-starttime))    
