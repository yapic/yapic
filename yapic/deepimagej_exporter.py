from yapic.session import Session
from yapic.networks import unet_2d
import os

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

        self.save_dir = os.path.dirname(save_path)
        msg = '{} does not exist'.format(self.save_dir)
        assert os.path.isdir(self.save_dir), msg

        self.s = Session()
        self.s.load_prediction_data(example_image_path, 'some/path')
        self.s.load_model(model_path)

        msg = 'model is not unet_2d, cannot be exported to deepimagej'
        assert self.is_model_unet_2d(), msg

        self.model_reshaped = None

    def is_model_unet_2d(self):
        return self.s.model.count_params() == 32424323

    def reshape_unet_2d(self, size='middle'):
        print(size)
        if size == 'small':
            shape_xy = 112
        elif size == 'middle':
            print('yes')
            shape_xy = 224
        elif size == 'large':
            shape_xy = 448
        else:
            shape_xy = 112
        print(shape_xy)
        N_classes = self.s.model.output_shape[-1]
        N_channels = self.s.model.input_shape[-1]

        self.model_reshaped = unet_2d.build_network(
                             N_classes,
                             (N_channels, 1, shape_xy, shape_xy),
                             squeeze=True,
                             padding='same')

        self.model_reshaped.set_weights(self.s.model.get_weights())
