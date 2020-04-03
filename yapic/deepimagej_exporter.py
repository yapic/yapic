from yapic.session import Session
from yapic.networks import unet_2d
import os
import tensorflow as tf
from keras import backend as K
import time
# from tensorflow.keras import backend as K


class DeepimagejExporter(object):
    '''
    A DeepImageJ exporter provides methods to deploy Keras models
    trained with YAPiC to the ImageJ plugin DeepImageJ.

    Parameters
    ----------
    model_path : string
        Path to Keras model in h5 format.
    save_path : string
        Path to directory where the exported model and metadata is saved.
    example_image_path: string
        Path to example input image in tif format.
    '''

    def __init__(self, model_path, save_path, example_image_path):

        save_dir = os.path.dirname(save_path)
        msg = '{} does not exist'.format(save_dir)
        assert os.path.isdir(save_dir), msg
        self.save_path = save_path

        self.s = Session()
        self.s.load_prediction_data(example_image_path, 'some/path')
        self.s.load_model(model_path)

        msg = 'model is not unet_2d, cannot be exported to deepimagej'
        assert self._is_model_unet_2d(), msg

        self.model_reshaped = None
        self.metadata = None

    def _is_model_unet_2d(self):
        return self.s.model.name == 'unet_2d'
        # return self.s.model.count_params() == 32424323

    def _reshape_unet_2d(self, size='middle'):

        if size == 'small':
            shape_xy = 112
        elif size == 'middle':
            shape_xy = 224
        elif size == 'large':
            shape_xy = 368
        else:
            shape_xy = 112

        N_classes = self.s.model.output_shape[-1]
        N_channels = self.s.model.input_shape[-1]

        self.model_reshaped = unet_2d.build_network(
                             N_classes,
                             (N_channels, 1, shape_xy, shape_xy),
                             squeeze=True,
                             padding='same')

        self.model_reshaped.set_weights(self.s.model.get_weights())

    def _update_metadata(self,
                         name='my_model',
                         author='n/a',
                         url='http://',
                         credit='n.a',
                         version='n.a',
                         reference='n/a'):

        if self.model_reshaped is None:
            return
        if self.metadata is None:
            self.metadata = {}

        self.metadata['name'] = name
        self.metadata['author'] = author
        self.metadata['url'] = url
        self.metadata['credit'] = credit
        self.metadata['version'] = version
        self.metadata['reference'] = reference

        date_format = '%a %b %d %H:%M:%S %Z %Y'
        self.metadata['date'] = time.strftime(date_format, time.localtime())

        N_channels = self.model_reshaped.input_shape[-1]
        size_xy = self.model_reshaped.input_shape[2]

        self.metadata['input_tensor_dimensions'] = (-1,
                                                    size_xy,
                                                    size_xy,
                                                    N_channels)
        self.metadata['patch_size'] = size_xy
        self.metadata['padding'] = 10 # fixme

        # metadata = {'name': 'my_model',
        #             'author': 'n/a',
        #             'url': 'http://',
        #             'credit': 'n/a',
        #             'version': 'n/a',
        #             'reference': 'n/a',
        #             'date': 'Tue Mar 31 17:18:06 CEST 2020',
        #             'test_image_size_xy': (512, 512),
        #             'input_tensor_dimensions': (-1, 112, 112, 3),
        #             'patch_size': (112),
        #             'padding': 10}

    def _export_as_tensorflow_model(self):

        model = self.model_reshaped
        builder = tf.saved_model.builder.SavedModelBuilder(self.save_path)

        signature = tf.saved_model.signature_def_utils.predict_signature_def(
                        inputs={'input':  model.input},
                        outputs={'output': model.output})

        signature_def_map = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}

        builder.add_meta_graph_and_variables(
            K.get_session(),
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map)
        builder.save()
