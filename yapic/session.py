from yapic.network import make_model, load_keras_model
import yapic.utils as ut
import numpy as np
from yapic_io.training_batch import TrainingBatch
from yapic_io.prediction_batch import PredictionBatch
from yapic_io.connector import io_connector
from yapic_io.dataset import Dataset
import tensorflow.keras as keras
import sys
from tensorflow.python.framework.tensor_shape import Dimension

import logging
logger = logging.getLogger(__name__)


class Session(object):
    '''
    A session is used for training a model with a connected dataset (pixels
    and labels) or for predicting connected data (pixels) with an already
    trained model.




    Parameters
    ----------
    data : yapic_io.TrainingBatch
        Connector object for binding pixel and label data
    '''

    def __init__(self):

        self.dataset = None
        self.model = None
        self.data = None
        self.data_val = None
        self.history = None
        self.data_predict = None
        self.log_filename = None

        self.output_tile_size_zxy = None
        self.padding_zxy = None

    def load_training_data(self, image_path, label_path):
        '''
        Connect to a training dataset.

        Parameters
        ----------
        image_path : string
            Path to folder with tiff images.
        label_path : string
            Path to folder with label tiff images or path to ilastik project
            file (.ilp file).
        '''

        self.dataset = Dataset(io_connector(image_path, label_path))

        msg = '\n\nImport taining dataset:\n{}\n'.format(
            self.dataset.pixel_connector.__repr__())
        sys.stdout.write(msg)

    def load_prediction_data(self, image_path, save_path):
        '''
        Connect to a prediction dataset.

        Parameters
        ----------
        image_path : string
            Path to folder with tiff images to predict.
        save_path : string
            Path to folder for saving prediction images.
        '''

        self.dataset = Dataset(io_connector(image_path,
                                            '/tmp/this_should_not_exist',
                                            savepath=save_path))
        msg = '\n\nImport dataset for prediction:\n{}\n'.format(
            self.dataset.pixel_connector.__repr__())
        sys.stdout.write(msg)

    def make_model(self, model_name, input_tile_size_zxy):
        '''
        Initialzes a neural network and sets tile sizes of the data connector
        accordingly to model input/output shapes.

        Parameters
        ----------
        model_name : string
            Either 'unet_2d' or 'unet_2p5d'
        input_tile_size_zxy: (nr_zslices, nr_x, nr_y)
            Input shape of the model. Large input shapes require large memory
            for used GPU hardware. For 'unet_2d', nr_zslices has to be 1.
        '''

        sys.stdout.write('\n\nInitialize model {}\n'.format(model_name))
        assert len(input_tile_size_zxy) == 3
        nr_channels = self.dataset.image_dimensions(0)[0]
        input_size_czxy = [nr_channels] + list(input_tile_size_zxy)
        n_classes = len(self.dataset.label_values())

        self.model = make_model(model_name, n_classes, input_size_czxy)

        output_tile_size_zxy = self.model.output_shape[-4:-1]

        output_tile_size_zxy = [v.value
                                if isinstance(v, Dimension) else v
                                for v in output_tile_size_zxy]

        self._configure_minibatch_data(input_tile_size_zxy,
                                       output_tile_size_zxy)

    def load_model(self, model_filepath):
        '''
        Import a Keras model in hfd5 format.

        Parameters
        ----------
        model_filepath : string
            Path to .h5 model file
        '''

        model = load_keras_model(model_filepath)

        n_classes_model = model.output_shape[-1]
        output_tile_size_zxy = model.output_shape[-4:-1]
        n_channels_model = model.input_shape[-1]
        input_tile_size_zxy = model.input_shape[-4:-1]

        n_classes_data = len(self.dataset.label_values())
        n_channels_data = self.dataset.image_dimensions(0)[0]

        msg = ('nr of model classes ({}) and data classes ({}) '
               'is not equal').format(n_classes_model, n_classes_data)
        if n_classes_data > 0:
            assert n_classes_data == n_classes_model, msg

        msg = ('nr of model channels ({}) and iamge channels ({}) '
               'is not equal').format(n_channels_model, n_channels_data)
        assert n_channels_data == n_channels_model, msg

        self._configure_minibatch_data(input_tile_size_zxy,
                                       output_tile_size_zxy)
        self.model = model

    def _configure_minibatch_data(self,
                                  input_tile_size_zxy,
                                  output_tile_size_zxy):
        padding_zxy = tuple(((np.array(input_tile_size_zxy) -
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
        '''
        Splits the dataset into a training fraction and a validation fraction.

        Parameters
        ----------
        valfraction : float
            Approximate fraction of validation data. Has to be between 0 and 1.
        '''
        msg = ('\nConfiuring validation dataset '
               '({} validation data, {} training data)').format(
                   valfraction,
                   1 - valfraction)
        sys.stdout.write(msg)
        if self.data_val is not None:
            logger.warning('skipping define_validation_data: already defined')
            return None

        if self.data is None:
            logger.warning('skipping, data not defined yet')
            return None
        self.data.remove_unlabeled_tiles()
        self.data_val = self.data.split(valfraction)

    def train(self, max_epochs=3000, steps_per_epoch=24, log_filename=None,
              model_filename='model.h5'):
        '''
        Starts a training run.

        Parameters
        ----------
        max_epochs : int
            Number of epochs.
        steps_per_epoch : int
            Number of training steps per epoch.
        log_filename : string
           Path to the csv file for logging loss and accuracy.
        model_filename : string
           Path to h5 keras model file


        Notes
        -----
        Validation is executed once each epoch.
        Logging to csv file is executed once each epoch.
        '''

        callbacks = []

        if self.data_val:
            save_model_callback = keras.callbacks.ModelCheckpoint(
                                        model_filename,
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True)
        else:
            save_model_callback = keras.callbacks.ModelCheckpoint(
                                        model_filename,
                                        monitor='loss',
                                        verbose=0,
                                        save_best_only=True)
        callbacks.append(save_model_callback)

        if log_filename:
            callbacks.append(keras.callbacks.CSVLogger(log_filename,
                                                       separator=',',
                                                       append=False))

        training_data = ((mb.pixels(), mb.weights())
                         for mb in self.data)

        validation_steps_per_epoch = steps_per_epoch
        if self.data_val:
            validation_data = ((mb.pixels(), mb.weights())
                               for mb in self.data_val)
        else:
            validation_data = None
            validation_steps_per_epoch = None

        self.history = self.model.fit_generator(
                            training_data,
                            validation_data=validation_data,
                            epochs=max_epochs,
                            validation_steps=validation_steps_per_epoch,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks,
                            workers=0)

        return self.history

    def predict(self, multichannel=False):
        data_predict = PredictionBatch(self.dataset,
                                       2,
                                       self.output_tile_size_zxy,
                                       self.padding_zxy)
        data_predict.set_normalize_mode('local')
        data_predict.set_pixel_dimension_order('bzxyc')

        if multichannel:
            data_predict.multichannel_output_on()
        else:
            data_predict.multichannel_output_off()
        print('multichannel output: {}'.format(data_predict.multichannel))

        for item_nr, item in enumerate(data_predict):
            msg = ('Writing probability map tile'
                   ' {} of {}...\n'.format(item_nr+1,
                                           len(data_predict)))
            sys.stdout.write(msg)
            sys.stdout.flush()
            result = self.model.predict(item.pixels())
            item.put_probmap_data(result)

        sys.stdout.write('Writing probability maps finished.\n')

    def set_augmentation(self, augment_string):
        '''
        Define data augmentation settings for model training.

        Parameters
        ----------
        augment_string : string
            Choose 'flip' and/or 'rotate' and/or 'shear'.
            Use '+' to specify multiple augmentations (e.g. flip+rotate).
        '''

        if self.data is None:
            logger.warning(
                'could not set augmentation to {}. Run make_model() first')
            return

        ut.handle_augmentation_setting(self.data, augment_string)

    def set_normalization(self, norm_string):
        '''
        Set pixel normalization scope.

        Parameters
        ----------
        norm_string : string
            For minibatch-wise normalization choose 'local_z_score' or 'local'.
            For global normalization use global_<min>+<max>
            (e.g. 'global_0+255' for 8-bit images and 'global_0+65535' for
            16-bit images).
            Choose 'off' to deactivate.
        '''

        if self.data is None:
            logger.warning(
                'could not set normalizarion to {}. Run make_model() first')
            return

        ut.handle_normalization_setting(self.data, norm_string)
