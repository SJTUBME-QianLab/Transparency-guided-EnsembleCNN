from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np


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

def get_full_VGG_classifier():
    inputs = Input((218,182,1))
    x = inputs


    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if TRAIN_CLASSIFY_USE_BN:
        x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = GlobalMaxPooling2D()(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=0.005), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def classify_train():
    print('start classify_train')


    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    imgs_train = np.load('train_data.npy')#加载训练集数据
    labels_train = np.load('train_label.npy')
    
    mean = np.mean(imgs_train)
    std = np.mean(imgs_train)
    
    imgs_train -= mean
    imgs_train /= std

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model= get_simplified_VGG_classifier()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    ##tb_cb = TensorBoard(log_dir=log_filepath, write_images=False, histogram_freq=1, write_graph=True)

    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(imgs_train, labels_train, batch_size=50, epochs=2, verbose=1, shuffle=True, validation_split=0.01,callbacks=[model_checkpoint])  
    
    print('-'*30)
    print('Loading test data...')
    print('-'*30)
    imgs_test = np.load('test_data.npy')#加载测试集数据
    #imgs_test = imgs_test.reshape(1134,218,182,1)
    imgs_test -= mean
    imgs_test /= std 

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting sorts on test data...')
    print('-'*30)
    imgs_predict_test = model.predict(imgs_test, batch_size=50, verbose=1)
    np.save('imgs_predict_test.npy', imgs_predict_test)


if __name__ == '__main__':
    classify_train()


