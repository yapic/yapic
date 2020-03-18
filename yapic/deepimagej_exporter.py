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
        assert os.path.isdir(save_path)
        self.s = Session()
        self.s.load_prediction_data(example_image_path, 'some/path')
        self.s.load_model(model_path)
