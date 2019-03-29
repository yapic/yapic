import keras

filter_size = (3, 3)
def contract(net, n_filters):
    net = keras.layers.Conv2D(n_filters,
                              filter_size,
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_first')(net)

    net = keras.layers.Conv2D(n_filters,
                              filter_size,
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_first')(net)
    to_concat = net

    net = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                    data_format='channels_first')(net)
    return net, to_concat

def expand(net, to_concat, n_filters):
    net = keras.layers.UpSampling2D(size=(2, 2),
                                    data_format='channels_first')(net)
    net = keras.layers.Conv2D(n_filters,
                              (2, 2),
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_first')(net)
    net = keras.layers.Conv2D(n_filters,
                              (2, 2),
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_first')(net)
    cropping = int(int(to_concat.shape[-1] - net.shape[-1])/2)
    cropped = keras.layers.Cropping2D(cropping=cropping,
                                      data_format='channels_first')(to_concat)
    net = keras.layers.Concatenate(axis=1)([net, cropped])

    net = keras.layers.Conv2D(n_filters,
                              (3, 3),
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_first')(net)
    net = keras.layers.Conv2D(n_filters,
                              (3, 3),
                              activation='relu',
                              kernel_initializer='glorot_normal',
                              data_format='channels_first')(net)

    return net


N_channels = 3
size_xy = 572

filter_size = (3, 3)

net = keras.layers.Input(shape=(N_channels, 1, size_xy, size_xy))
input_net = net

net = keras.layers.Reshape((N_channels, size_xy, size_xy))(net)

net, to_concat_1 = contract(net, 64)
net, to_concat_2 = contract(net, 128)
net, to_concat_3 = contract(net, 256)
net, to_concat_4 = contract(net, 512)
net, to_concat_5 = contract(net, 1024)
net = expand(to_concat_5, to_concat_4, 512)
net = expand(net, to_concat_3, 256)
net = expand(net, to_concat_2, 128)
net = expand(net, to_concat_1, 64)
