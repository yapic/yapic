import logging
import keras

logger = logging.getLogger(__name__)

def build_network(N_channels, N_class_list):
    '''
    Builds the original U-Net by Ronneberger et al. (2015) for 2d images
    '''
    logger.debug('Building network with {} classes'.format(N_class_list))

    # net = lasagne.layers.InputLayer(input_var=input_var,
    #                                 shape=(None, N_channels, 1,572,572))


    size_xy = 572
    net = keras.layers.Input(shape=(N_channels, 1, size_xy, size_xy))
    input_net = net

    # drop Z dimension [B, C, Z, X, Y]
    #net = lasagne.layers.ReshapeLayer(net, ([0], [1], [3], [4]))
    net = keras.layers.Reshape((N_channels, size_xy, size_xy))(net)

    filter_list = [64, 128, 256, 512]
    filter_size_list = [(3,3),
                        (3,3),
                        (3,3),
                        (3,3)]

    # downscaling
    layer_list = []
    for num_filters, filter_size in zip(filter_list, filter_size_list): # layers
        for _ in range(2):
            net = keras.layers.Conv2D(num_filters,
                                      filter_size,
                                      activation='relu',
                                      kernel_initializer='glorot_normal',
                                      data_format='channels_first')(net)
        layer_list.append(net)

        # net = lasagne.layers.MaxPool2DLayer(net, pool_size=(2,2))
        net = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        data_format='channels_first')(net)

    # lowest layer
    for _ in range(2):
        net = keras.layers.Conv2D(1024,
                                  (3, 3),
                                  activation='relu',
                                  kernel_initializer='glorot_normal',
                                  data_format='channels_first')(net)
    # for _ in range(2):
    #     net = keras.layers.Conv2DTranspose(1024,
    #                                        (3, 3),
    #                                        # strides=(2, 2),
    #                                        activation='relu',
    #                                        kernel_initializer='glorot_normal',
    #                                        data_format='channels_first')(net)

    # upscaling
    print(layer_list)
    print(net)
    for layer, num_filters, filter_size in zip(*map(reversed, [layer_list, filter_list, filter_size_list])): # layers
        #net = lasagne.layers.Upscale2DLayer(net, 2)
        # net = lasagne.layers.Deconv2DLayer(net,
        #                                    num_filters=num_filters,
        #                                    filter_size=filter_size,
        #                                    stride=(2,2))

        for _ in range(2):
            net = keras.layers.Conv2DTranspose(num_filters,
                                               filter_size,
                                               # strides=(2, 2),
                                               activation='relu',
                                               kernel_initializer='glorot_normal',
                                               data_format='channels_first')(net)
        net = keras.layers.UpSampling2D(size=(2, 2),
                                        data_format='channels_first')(net)
        # net = lasagne.layers.ConcatLayer([net, layer], axis=1, cropping=[None, None, 'center', 'center'])
        net = keras.layers.Concatenate(axis=1)([net, layer])

        for _ in range(2):
            net = keras.layers.Conv2D(num_filters,
                                      filter_size,
                                      activation='relu',
                                      kernel_initializer='glorot_normal',
                                      data_format='channels_first')(net)


    output_channels = []
    for N_classes in N_class_list:
        ch = keras.layers.Conv2D(N_classes,
                                 (1, 1),
                                  activation='softmax',
                                  data_format='channels_first')(net)
        output_channels.append(ch)

    # net = lasagne.layers.ConcatLayer(output_channels, axis=1)
    net = keras.layers.Concatenate(axis=1)(output_channels)
    print(net)
    #net = lasagne.layers.SliceLayer(net, indices=slice(0,-1), axis=-1)
    #net = lasagne.layers.SliceLayer(net, indices=slice(0,-1), axis=-2)

    # add Z dimension again
    #net = lasagne.layers.ReshapeLayer(net, ([0], [1], -1, [2], [3]))

    return net
