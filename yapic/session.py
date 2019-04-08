import logging
import os
from yapic.network import make_model
import numpy as np
from yapic_io.training_batch import TrainingBatch
from yapic_io.prediction_batch import PredictionBatch
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.dataset import Dataset
import csv
from time import localtime, strftime
import keras
from keras.callbacks import TensorBoard

# tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
#                           write_graph=True, write_images=False)

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)


class Session(object):
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
        self.data_predict = None
        self.log_filename = None

        self.output_tile_size_zxy = None
        self.padding_zxy = None

    def make_model(self, model_name, input_tile_size_zxy):

        assert len(input_tile_size_zxy) == 3
        nr_channels = self.dataset.image_dimensions(0)[0]
        print('nr_channels: {}'.format(nr_channels))
        input_size_czxy = [nr_channels] + list(input_tile_size_zxy)
        n_classes = len(self.dataset.label_values())

        self.model = make_model(model_name, n_classes, input_size_czxy)

        output_tile_size_zxy = self.model.output_shape[-4:-1]
        padding_zxy = tuple(((np.array(input_tile_size_zxy) - \
                            np.array(output_tile_size_zxy))/2).astype(np.int))

        self.data_val = None
        self.data = TrainingBatch(self.dataset,
                                  output_tile_size_zxy,
                                  padding_zxy=padding_zxy)
        self.data.set_normalize_mode('local')
        self.data.set_pixel_dimension_order('bzxyc')
        next(self.data)
        self.output_tile_size_zxy = output_tile_size_zxy
        self.padding_zxy = padding_zxy

    def define_validation_data(self, valfraction):
        if self.data_val is not None:
            logger.warning('skipping define_validation_data: already defined')
            return None

        if self.data is None:
            logger.warning('skipping, data not defined yet')
            return None
        # self.data.remove_unlabeled_tiles()
        self.data_val = self.data.split(valfraction)

    def train(self, max_epochs=3000, steps_per_epoch=24, log_filename=None):

        callbacks = []

        if log_filename:
            callbacks.append(keras.callbacks.CSVLogger(log_filename,
                                                   separator=',',
                                                   append=False))

        training_data = ((mb.pixels(), mb.weights())
                         for mb in self.data)

        if self.data_val:
            validation_data = ((mb.pixels(), mb.weights())
                               for mb in self.data_val)
        else:
            validation_data = None

        self.history = self.model.fit_generator(
                            training_data,
                            validation_data=validation_data,
                            epochs=max_epochs,
                            validation_steps=2,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks,
                            workers=0)

        return self.history

    def predict(self):
        data_predict = PredictionBatch(self.dataset,
                                       2,
                                       self.output_tile_size_zxy,
                                       self.padding_zxy)
        data_predict.set_normalize_mode('local')
        data_predict.set_pixel_dimension_order('bzxyc')

        for item in data_predict:
            result = self.model.predict(item.pixels())
            item.put_probmap_data(result)




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
