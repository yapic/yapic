from unittest import TestCase
from yapic.deepimagej_exporter import DeepimagejExporter
from yapic.session import Session
import os

base_path = os.path.dirname(__file__)


def train_test_model_1():
    img_path = os.path.join(
        base_path,
        '../test_data/shapes/pixels/*')
    label_path = os.path.join(
        base_path,
        '../test_data/shapes/labels.ilp')
    savepath = os.path.join(
        base_path,
        '../test_data/tmp')

    os.makedirs(savepath, exist_ok=True)

    t = Session()
    t.load_training_data(img_path, label_path)
    t.make_model('convnet_for_unittest', (1, 100, 100))

    model_filename = os.path.join(savepath, 'model.h5')
    t.train(max_epochs=2,
            steps_per_epoch=4,
            log_filename=os.path.join(savepath, 'log.csv'),
            model_filename=model_filename)

    return model_filename





class TestDeepimagejExporter(TestCase):

    def test_init_deepimagej_exporter(self):

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels/pixels_1.tiff'))
        
        model_path = train_test_model_1()
        print('model_path: {}'.format(model_path))
        save_path = ''

        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)
        assert False
