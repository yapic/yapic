import logging

import keras

logger = logging.getLogger(__name__)


def build_network(N_channels, N_class_list):

    logger.debug('Building network with {} classes'.format(N_class_list))
    assert len(N_class_list) == 1

    net = keras.layers.Input(shape=(N_channels, 1, 1, 3))
    input_net = net

    # drop Z dimension [C, Z, X, Y]
    net = keras.layers.Reshape((1, 1, 3))(net)

    for _ in range(1):
        net = keras.layers.Conv2D(5,
                                  (1, 3),
                                  activation='relu',
                                  kernel_initializer='glorot_normal',
                                  data_format='channels_first')(net)

    net = keras.layers.Conv2D(N_class_list[0],
                              (1, 1),
                              activation='softmax',
                              data_format='channels_first')(net)

    net = keras.layers.Reshape((1, 1, 1, -1))(net)

    return keras.models.Model(inputs=input_net, outputs=net)
