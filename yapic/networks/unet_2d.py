from tensorflow import keras
import logging
import os
import math

logger = logging.getLogger(os.path.basename(__file__))


def convolve(net, n_filters, filter_size, padding):
    '''
    double convolution step
    '''
    net = keras.layers.Conv2D(n_filters,
                              filter_size,
                              padding=padding,
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_last')(net)
    net = keras.layers.Conv2D(n_filters,
                              filter_size,
                              padding=padding,
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_last')(net)
    return net


def contract(net, n_filters, padding):
    net = convolve(net, n_filters, (3, 3), padding)
    to_concat = net

    net = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    data_format='channels_last')(net)
    return net, to_concat


def expand(net, to_concat, n_filters, padding):
    net = keras.layers.UpSampling2D(size=(2, 2),
                                    data_format='channels_last')(net)
    net = convolve(net, n_filters, (2, 2), padding)

    shape_diff = to_concat.shape[-2] - net.shape[-2]
    cropping = int(math.floor(int(shape_diff)/2.))

    cropped = keras.layers.Cropping2D(cropping=cropping,
                                      data_format='channels_last')(to_concat)

    net = keras.layers.Concatenate(axis=-1)([net, cropped])
    net = convolve(net, n_filters, (3, 3), padding)

    return net


def build_network(N_classes, input_size_czxy, squeeze=False, padding='valid'):
    '''
    Builds the original U-Net by Ronneberger et al. (2015) for 2d images
    '''
    logger.debug('Building network with {} classes'.format(N_classes))

    assert len(input_size_czxy) == 4
    N_channels = input_size_czxy[0]
    size_z = input_size_czxy[1]
    size_xy = input_size_czxy[3]

    assert N_classes >= 2
    assert N_channels >= 1
    assert size_z == 1
    assert input_size_czxy[2] == input_size_czxy[3]

    if squeeze:
        net = keras.layers.Input(shape=(size_xy, size_xy, N_channels))
        input_net = net
    else:
        net = keras.layers.Input(shape=(size_z, size_xy, size_xy, N_channels))
        input_net = net
        net = keras.layers.Reshape((size_xy, size_xy, N_channels))(net)

    net, to_concat_1 = contract(net, 64, padding)
    net, to_concat_2 = contract(net, 128, padding)
    net, to_concat_3 = contract(net, 256, padding)
    net, to_concat_4 = contract(net, 512, padding)
    net, to_concat_5 = contract(net, 1024, padding)
    net = expand(to_concat_5, to_concat_4, 512, padding)
    net = expand(net, to_concat_3, 256, padding)
    net = expand(net, to_concat_2, 128, padding)
    net = expand(net, to_concat_1, 64, padding)

    net = keras.layers.Conv2D(N_classes,
                              (1, 1),
                              activation='softmax',
                              data_format='channels_last')(net)

    size_xy_out = int(net.shape[-2])
    if not squeeze:
        net = keras.layers.Reshape((1,
                                    size_xy_out,
                                    size_xy_out,
                                    N_classes))(net)
    model = keras.models.Model(inputs=input_net, outputs=net, name='unet_2d')

    return model
