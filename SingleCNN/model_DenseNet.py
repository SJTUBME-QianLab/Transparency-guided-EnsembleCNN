from keras.models import Model
from keras.layers import Input, Conv2D, Dense, BatchNormalization, Concatenate, Dropout, AveragePooling2D, GlobalAveragePooling2D, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np

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

    return model


def classify_train():
    #print('start classify_train')


    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    imgs_train = np.load('train_data.npy')#加载训练集数据
    labels_train = np.load('train_label.npy')
    
    mean = np.mean(imgs_train)
    std = np.mean(imgs_train)
    
    #imgs_train -= mean
    #imgs_train /= std

    #print('-'*30)
    #print('Creating and compiling model...')
    #print('-'*30)
    model= get_DenseNet_classifier()
    #model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    ##tb_cb = TensorBoard(log_dir=log_filepath, write_images=False, histogram_freq=1, write_graph=True)

    
    #print('-'*30)
    #print('Fitting model...')
    #print('-'*30)

    #model.fit(imgs_train, labels_train, batch_size=30, epochs=30, verbose=1, shuffle=False, validation_split=0.01,callbacks=[model_checkpoint])  
    
    print('-'*30)
    print('Loading test data...')
    print('-'*30)
    imgs_test = np.load('test_data.npy')#加载测试集数据
    ##imgs_test = imgs_test.reshape(1134,218,182,1)
    imgs_test -= mean
    imgs_test /= std 

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting sorts on test data...')
    print('-'*30)
    imgs_predict_test = model.predict(imgs_test, batch_size=30, verbose=1)
    np.save('imgs_predict_test.npy', imgs_predict_test)


if __name__ == '__main__':
    classify_train()
