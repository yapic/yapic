from tensorflow import keras


def build_network(N_classes, input_size_czxy):
    '''
    builds a small model suitable for running on cpu
    '''
    assert len(input_size_czxy) == 4

    N_channels = input_size_czxy[0]
    size_z = input_size_czxy[1]
    size_xy = input_size_czxy[3]

    assert N_classes >= 2
    assert N_channels >= 1
    assert size_z == 1
    assert input_size_czxy[2] == input_size_czxy[3]

    net = keras.layers.Input(shape=(size_z, size_xy, size_xy, N_channels))
    input_net = net

    net = keras.layers.Reshape((size_xy, size_xy, N_channels))(net)

    n_convolutions = 6
    n_filters = 50
    filter_size = (3, 3)
    for _ in range(n_convolutions):
        net = keras.layers.Conv2D(n_filters,
                                  filter_size,
                                  activation='relu',
                                  kernel_initializer='glorot_normal',
                                  data_format='channels_last')(net)

    net = keras.layers.Conv2D(N_classes,
                              (1, 1),
                              activation='softmax',
                              data_format='channels_last')(net)

    size_xy_out = int(net.shape[-2])

    net = keras.layers.Reshape((1, size_xy_out, size_xy_out, N_classes))(net)
    model = keras.models.Model(inputs=input_net, outputs=net)

    return model
