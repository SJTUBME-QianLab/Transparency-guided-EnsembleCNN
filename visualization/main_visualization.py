import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob as gb
from keras import backend as K
from keras.models import Model
from keras import models
from keras.layers import Input, Dense,MaxPooling2D, BatchNormalization, Concatenate, Dropout, AveragePooling2D, GlobalAveragePooling2D, Conv2D, Add, Flatten, GlobalMaxPooling2D, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping



'''#VGG16
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

    return model'''

'''#Resnet
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

    return model'''

#DenseNet
DENSE_NET_BLOCKS = 3
DENSE_NET_BLOCK_LAYERS = 5
DENSE_NET_INITIAL_CONV_DIM = 8
DENSE_NET_GROWTH_RATE = DENSE_NET_INITIAL_CONV_DIM // 2
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

model= get_DenseNet_classifier()
model.load_weights('DenseNet_weights.h5')

model.summary()

#img_o =np.load('data.npy')##import the original medical image
#input_img_data=img_o[0]
#input_img_data=input_img_data.reshape(1,218,182,1)

#change the name of layer you want to show here————>>>
#layer_name='conv2d_4'

#layer_output = model.get_layer(layer_name).output


'''####################################################
#visualization layer by layer
####################################################

# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_output)
activations = activation_model.predict(input_img_data)
#plt.matshow(activations, cmap='viridis')
#plt.show()

images_per_row = 16

for layer_activation in activations[:,:,:,:]:
    
    n_features = layer_activation.shape[-1]

    size1 = layer_activation.shape[0]
    size2 = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size1 * n_cols, images_per_row * size2))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[:, :,col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size1:(col + 1) * size1, row * size2:(row + 1) * size2] = channel_image

    # Display the grid
    #scale = 1. / size
    plt.figure()
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, cmap='gray')

plt.axis('off')    
plt.show()

n= activations[:,:,:,17]

n = n.reshape(109,91)

plt.imshow(n, cmap='gray')

plt.axis('off')    
plt.show()'''


'''####################################################
#show the covolutional kernel
####################################################

filter_index = 0

loss = K.mean(layer_output[:, :, :, filter_index])

grads = K.gradients(loss, model.input)[0]

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

iterate = K.function([model.input], [loss, grads])

loss_value, grads_value = iterate([np.zeros((1, 218, 182, 1))])

#input_img_data = np.random.random((1, 218, 182, 1)) * 20 + 128

step = 1.  # this is the magnitude of each gradient update
for i in range(40):
    # Compute the loss value and gradient value
    loss_value, grads_value = iterate([input_img_data])
    # Here we adjust the input image in the direction that maximizes the loss
    input_img_data += grads_value * step

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    #input_img_data = np.random.random((1, 218, 182, 1)) * 20 + 128.
    img=np.load('data.npy')
    input_img_data=img[0]
    input_img_data=input_img_data.reshape(1,218,182,1)

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

#sh=generate_pattern('conv2d_10', 0)
#sh=sh.reshape(218,182)

size1 = 218
size2= 182
margin = 5

# This a empty (black) image where we will store our results.
results = np.zeros((4 * size1 + 3 * margin, 4 * size2 + 3 * margin))

for i in range(4):
    for j in range(4):  # iterate over the columns of our results grid
        filter_img = generate_pattern(layer_name, j + (i * 4))
        filter_img = filter_img.reshape(218,182)
        horizontal_start = i * size1 + i * margin
        horizontal_end = horizontal_start + size1
        vertical_start = j * size2 + j * margin
        vertical_end = vertical_start + size2
        results[horizontal_start: horizontal_end, vertical_start: vertical_end] = filter_img


plt.figure()
plt.imshow(results,cmap='gray')
plt.axis('off')
plt.show()'''

#plt.imshow(sh,'gray')
#plt.axis('off')
#plt.show()



'''#######################################################
#show heatmap and map it back
#######################################################

for m in range(1348):
    input_img_data = img_o[m]
    input_img_data = input_img_data.reshape(1,218,182,1)
    preds = model.predict(input_img_data, batch_size=1, verbose=1)
    a=np.argmax(preds[0])
    output = model.output[:, a]
    last_conv_layer = model.get_layer('concatenate_15')
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([input_img_data])
    for i in range(68):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = cv2.imread('data/image/' + str(m+1) + '.jpg')
        heatmap = cv2.resize(heatmap, (input_img_data.shape[2], input_img_data.shape[1]))
        # We convert the heatmap to RGB
        heatmap = np.uint8(255*heatmap)
        # We apply the heatmap to the original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        ## 0.4 here is a heatmap intensity factor
        superimposed_img = heatmap * 0.4 + img
        ## Save the image to disk
        cv2.imwrite('V/DenseNetheat/' + str(m+1) + '.jpg',superimposed_img)
    #plt.imshow(heatmap)
    #plt.axis('off')
    #plt.colorbar()
    #plt.show()'''

    
