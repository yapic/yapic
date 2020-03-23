from unittest import TestCase
from yapic.deepimagej_exporter import DeepimagejExporter
from yapic.session import Session
import os

base_path = os.path.dirname(__file__)


def train_test_model_unet_2d():
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
    t.make_model('unet_2d', (1, 220, 220))

    model_filename = os.path.join(savepath, 'model_unet_2d.h5')
    t.train(max_epochs=2,
            steps_per_epoch=2,
            log_filename=os.path.join(savepath, 'log.csv'),
            model_filename=model_filename)

    return model_filename


def train_test_model_convnet():
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

    model_filename = os.path.join(savepath, 'model_convnet.h5')
    t.train(max_epochs=2,
            steps_per_epoch=2,
            log_filename=os.path.join(savepath, 'log.csv'),
            model_filename=model_filename)

    return model_filename


class TestDeepimagejExporter(TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = train_test_model_unet_2d()
        print('saved unet_2d as {}'.format(model_path))

        # model_path = train_test_model_convnet()
        # print('saved unet_2d as {}'.format(model_path))

    def test_is_model_unet_2d(self):

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels/pixels_1.tiff'))

        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model'))

        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d.h5')

        print('model_path: {}'.format(model_path))

        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)

        assert exp.is_model_unet_2d()

    # def test_is_model_unet_2d_2(self):
    #
    #     example_image_path = os.path.abspath(os.path.join(
    #         base_path,
    #         '../test_data/shapes/pixels/pixels_1.tiff'))
    #
    #     save_path = os.path.abspath(os.path.join(
    #         base_path,
    #         '../test_data/tmp/exported_model'))
    #
    #     model_path = os.path.join(
    #         base_path,
    #         '../test_data/tmp/model_convnet.h5')
    #
    #     print('model_path: {}'.format(model_path))
    #
    #     exp = DeepimagejExporter(model_path,
    #                              save_path,
    #                              example_image_path)
    #
    #     assert exp.is_model_unet_2d() is False
