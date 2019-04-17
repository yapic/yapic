from yapic.networks import unet_2d
import keras
import logging
import os
import math



def build_network(N_classes, input_size_czxy):

    N_channels = input_size_czxy[0]
    size_z = input_size_czxy[1]
    size_xy = input_size_czxy[3]

    assert size_z == 5

    input_size_z_layer = list(input_size_czxy)
    input_size_z_layer[1] = 1
    print(input_size_z_layer)
    print(input_size_czxy)
    z_layer = unet_2d.build_network(N_classes,
                                    (N_channels,
                                     1,
                                     size_xy,
                                     size_xy))
    print(z_layer.output_shape)

    input = keras.layers.Input(shape=(size_z, size_xy, size_xy, N_channels))

    input0 = keras.layers.Cropping3D(cropping=((0, 4), (0, 0), (0, 0)),
                            data_format='channels_last')(input)
    input1 = keras.layers.Cropping3D(cropping=((1, 3), (0, 0), (0, 0)),
                            data_format='channels_last')(input)
    input2 = keras.layers.Cropping3D(cropping=((2, 2), (0, 0), (0, 0)),
                            data_format='channels_last')(input)
    input3 = keras.layers.Cropping3D(cropping=((3, 1), (0, 0), (0, 0)),
                            data_format='channels_last')(input)
    input4 = keras.layers.Cropping3D(cropping=((4, 0), (0, 0), (0, 0)),
                            data_format='channels_last')(input)

    net = keras.layers.Concatenate(axis=1)([z_layer(input0),
                                             z_layer(input1),
                                             z_layer(input2),
                                             z_layer(input3),
                                             z_layer(input4)])
    print(net.shape)
    net = keras.layers.Conv3D(25,
                              (3,3,3),
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_last')(net)
    print(net.shape)
    net = keras.layers.Conv3D(N_classes,
                              (3,3,3),
                              activation='softmax',
                              kernel_initializer='glorot_normal',
                              data_format='channels_last')(net)
    print(net.shape)
    model = keras.models.Model(inputs=input, outputs=net)

    return model
