from yapic.networks import unet_2d
from tensorflow import keras


def build_network(N_classes, input_size_czxy):

    N_channels = input_size_czxy[0]
    size_z = input_size_czxy[1]
    size_xy = input_size_czxy[3]

    assert size_z == 5

    input_size_z_layer = list(input_size_czxy)
    input_size_z_layer[1] = 1

    input = keras.layers.Input(shape=(size_z, size_xy, size_xy, N_channels))

    # one 2d unet for each z-slice
    z_layer = unet_2d.build_network(N_classes,
                                    (N_channels,
                                     1,
                                     size_xy,
                                     size_xy))
    models_single_z_layer = []
    for crop_z in [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]:
        input_single_z_layer = keras.layers.Cropping3D(
                                        cropping=(crop_z, (0, 0), (0, 0)),
                                        data_format='channels_last')(input)
        models_single_z_layer.append(z_layer(input_single_z_layer))

    net = keras.layers.Concatenate(axis=1)(models_single_z_layer)

    net = keras.layers.Conv3D(25,
                              (3, 3, 3),
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_last')(net)

    net = keras.layers.Conv3D(N_classes,
                              (3, 3, 3),
                              activation='softmax',
                              kernel_initializer='glorot_normal',
                              data_format='channels_last')(net)

    model = keras.models.Model(inputs=input, outputs=net)

    return model
