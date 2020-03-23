from yapic.session import Session
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

        save_dir = os.path.dirname(save_path)
        assert os.path.isdir(save_dir), '{} does not exist'.format(save_dir)
        self.s = Session()
        self.s.load_prediction_data(example_image_path, 'some/path')
        self.s.load_model(model_path)

        msg = 'model is not unet_2d, cannot be exported to deepimagej'
        assert self.is_model_unet_2d(), msg

    def is_model_unet_2d(self):
        return self.s.model.count_params() == 32424323

    def reshape_unet_2d(self, size='middle'):

        if size=='small':
            shape_xy = 112
        if size=='middle':
            shape_xy = 224
        if size=='large':
            shape_xy = 448
        else:
            shape_xy = 112

        
        model_reshaped = unet_2d.build_network(N_classes,
                                              (N_channels, 1, small_xy, small_xy),
                                               squeeze=True, padding='same')
