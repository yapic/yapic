import logging
import os
from yapic.network import make_model
import numpy as np
from yapic_io.training_batch import TrainingBatch
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.dataset import Dataset

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)


class TrainingProject(object):
    '''
    Provides connectors to pixel data source and assigned weights for
    classifier training.
    Provides methods for getting image tiles and data augmentation for
    classifier training, as well as writing classifier output tile-by-tile
    to target images.



    Parameters
    ----------
    data : yapic_io.TrainingBatch
        Connector object for binding pixel and label data

    Notes
    -----
    Pixel data is loaded lazily to allow images of arbitrary size.
    Pixel data is cached in memory for repeated requests.
    '''

    def __init__(self, dataset):

        self.dataset = dataset
        self.model = None
        self.data = None
        self.data_val = None
        self.history = None

    def make_model(self, model_name, input_tile_size_zxy):

        assert len(input_tile_size_zxy) == 3
        nr_channels = self.dataset.image_dimensions(0)[0]
        input_size_czxy = [nr_channels] + list(input_tile_size_zxy)
        n_classes = len(self.dataset.label_values())

        self.model = make_model(model_name, n_classes, input_size_czxy)

        output_tile_size_zxy = self.model.output_shape[-3:]
        padding_zxy = tuple(((np.array(input_tile_size_zxy) - \
                            np.array(output_tile_size_zxy))/2).astype(np.int))
        print(input_tile_size_zxy)
        print(output_tile_size_zxy)
        print('padding_zxy: {}'.format(padding_zxy))

        self.data_val = None
        self.data = TrainingBatch(self.dataset,
                                  output_tile_size_zxy,
                                  padding_zxy=padding_zxy)
        next(self.data)

    def define_validation_data(self, valfraction):
        if self.data_val is not None:
            logger.warning('skipping define_validation_data: already defined')
            return None

        if self.data is None:
            logger.warning('skipping, data not defined yet')
            return None
        self.data_val = self.data.split(valfraction)

    def train(self, max_epochs=3000, workers=0):

        training_data = ((mb.pixels(), mb.weights()) for mb in self.data)
        self.history = self.model.fit_generator(training_data,
                                           validation_data=None,
                                           epochs=max_epochs,
                                           validation_steps=None,
                                           steps_per_epoch=12,
                                           callbacks=[],
                                           workers=workers)

        # val_loss = history.history['val_loss'][-1]
        # N_epochs = len(history.history['val_loss'])
        #
        # val_channel_accs = [history.history[key][-1]
        #                     for key in history.history.keys()
        #                     if key.startswith('val_') and key.endswith(
        #                                                      '_accuracy')]
        # val_accuracy = np.mean(val_channel_accs)
        #
        # return val_loss, val_accuracy, N_epochs
        return self.history





def load_shape_dataset():
    base_path = os.path.dirname(__file__)
    img_path = os.path.join(
        base_path,
        'test_data/shapes/pixels/*')
    label_path = os.path.join(
        base_path,
        'test_data/shapes/labels.ilp')
    c = IlastikConnector(img_path, label_path)
    return Dataset(c)




# def train(net, training_minibatch, validation_minibatch,
#           max_epochs=3000,
#           accuracy_threshold=0.97,
#           loss_threshold=1e-5,
#           threshold_epochs=2,
#           stop_fn=None,
#           validation_fn=None,
#           deterministic=False):
#     '''
#     Runs a training on a network
#
#     Returns:
#         (val_loss, val_accuracy, N_epochs) tuple
#         val_loss:     value of loss function for last validation run
#         val_accuracy: accuracy of last validation run in range 0 to 1
#         N_epochs:     Number of epochs performed in training
#
#     :param max_epochs: (int) Maximum number of epochs the training should run
#     :param accuracy_threshold: (float) Finish training if validation accuracy
#                                exceeds this value (in range 0 to 1) for
#                                `threshold_epochs` epochs.
#     :param loss_threshold: (float) Finish training if loss function for
#                                validation goes below this value for
#                                `threshold_epochs` epochs.
#     :param threshold_epochs: (int) Finish training only if thresholds are hold
#                                for this number of epochs in a row
#     :param stop_fn: function taking (train_loss, val_loss, val_accuracy_per_class)
#                     and returning true if the training should stop.
#                     (overwrites behaviour of `accuracy_threshold`, `loss_threshold`,
#                     and `threshold_epochs`).
#     :param validation_fn: callback function taking (network, train_loss, val_loss)
#                     as input. The function is executed evry time a validation step
#                     is performed.
#     :param deterministic: do not shuffle, produce deterministic results
#     '''
#     validation_steps = 5 # TODO
#     steps_per_epoch = 10 # TODO
#
#     if stop_fn is None:
#         class StopFunc():
#             thresh_epochs = 0
#
#             def stop_fn(self, train_loss, val_loss, val_accuracy_per_class):
#                 self.thresh_epochs += 1
#                 if val_loss > loss_threshold or np.mean(val_accuracy_per_class) < accuracy_threshold:
#                     self.thresh_epochs = 0
#
#                 return self.thresh_epochs == threshold_epochs
#         stop_fn = StopFunc().stop_fn
#
#     callbacks = [EarlyStopping(stop_fn)]
#     if validation_fn:
#         callbacks.append(UserDefinedCallback(validation_fn))
#
#     history = net.fit_generator(training_minibatch,
#         validation_data=validation_minibatch,
#         epochs=max_epochs,
#         validation_steps=validation_steps,
#         steps_per_epoch=steps_per_epoch,
#         callbacks=callbacks,
#         workers=0 if deterministic else 1,
#         )
#
#     val_loss = history.history['val_loss'][-1]
#     N_epochs = len(history.history['val_loss'])
#
#     val_channel_accs = [history.history[key][-1] for key in history.history.keys()
#                         if key.startswith('val_') and key.endswith('_accuracy')]
#     val_accuracy = np.mean(val_channel_accs)
#
#     return val_loss, val_accuracy, N_epochs
