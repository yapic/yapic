import numpy as np
import keras.optimizers
from keras import backend as K
from keras import metrics
import logging

logger = logging.getLogger(__name__)


def setup_network(network_name, N_classes, input_size_czxy):
    '''
    Builds up the network named `network_name` with `N_channels` input channels
    and `N_class_list` output classes.
    '''

    try:
        module_path = 'yapic.networks.{}'.format(network_name)
        root_mod = __import__(module_path)
    except ImportError:
        msg = 'Could not import network "{name}" (file networks/{name}.py)'
        raise ImportError(msg.format(name=network_name))

    network_module = getattr(root_mod.networks, network_name)
    network = network_module.build_network(N_classes, input_size_czxy)

    return network


def compile_model(network, learning_rate=1e-2, momentum=0.9):
    '''
    Compiles the network
    '''
    count_labels = lambda y: K.sum(K.cast(K.any(K.not_equal(y, 0), axis=-1),
                                   dtype=K.floatx()))
    correct_mean = lambda y: K.prod(K.cast(K.shape(y)[:-1],
                                    dtype=K.floatx())) / count_labels(y)

    def accuracy(y_true, y_pred):
        return metrics.categorical_accuracy(y_true,
                                            y_pred) * correct_mean(y_true)

    optimize = keras.optimizers.SGD(lr=learning_rate,
                                    momentum=momentum,
                                    nesterov=True)
    network.compile(optimizer=optimize,
                    loss='categorical_crossentropy',
                    metrics=[accuracy])

    return network


def make_model(network_name, N_classes, input_size_czxy,
               learning_rate=1e-2, momentum=0.9):
    logger.info(
        'Building network "{}" with input size {} (czxy) and {} classes...'
        .format(network_name, input_size_czxy, N_classes))
    net = setup_network(network_name, N_classes, input_size_czxy)

    net = compile_model(net, learning_rate=learning_rate, momentum=momentum)
    logger.debug('Network built with output size {}.'
                 .format(net.output_shape))
    return net
