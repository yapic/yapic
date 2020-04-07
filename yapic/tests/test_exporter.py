from unittest import TestCase
from yapic.deepimagej_exporter import DeepimagejExporter
from yapic.session import Session
import os
import shutil
import tensorflow as tf
import pytest

base_path = os.path.dirname(__file__)


def train_test_model_unet_2d_1channel_3classes():
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

    model_filename = os.path.join(savepath,
                                  'model_unet_2d_1channel_3classes.h5')
    t.train(max_epochs=2,
            steps_per_epoch=2,
            log_filename=os.path.join(savepath, 'log.csv'),
            model_filename=model_filename)

    return model_filename


def train_test_model_unet_2d_1channel_2classes():
    img_path = os.path.join(
        base_path,
        '../test_data/shapes/pixels/*')
    label_path = os.path.join(
        base_path,
        '../test_data/shapes/labels_2classes.ilp')
    savepath = os.path.join(
        base_path,
        '../test_data/tmp')

    os.makedirs(savepath, exist_ok=True)

    t = Session()
    t.load_training_data(img_path, label_path)
    t.make_model('unet_2d', (1, 220, 220))

    model_filename = os.path.join(savepath,
                                  'model_unet_2d_1channel_2classes.h5')
    t.train(max_epochs=2,
            steps_per_epoch=2,
            log_filename=os.path.join(savepath, 'log.csv'),
            model_filename=model_filename)

    return model_filename


def train_test_model_unet_2d_3channels_2classes():
    img_path = os.path.join(
        base_path,
        '../test_data/shapes/pixels_rgb/*')
    label_path = os.path.join(
        base_path,
        '../test_data/shapes/labels_2classes.ilp')
    savepath = os.path.join(
        base_path,
        '../test_data/tmp')

    os.makedirs(savepath, exist_ok=True)

    t = Session()
    t.load_training_data(img_path, label_path)
    t.make_model('unet_2d', (1, 220, 220))

    model_filename = os.path.join(savepath,
                                  'model_unet_2d_3channels_2classes.h5')
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

        #delete test artifacts
        savepath = os.path.join(
            base_path,
            '../test_data/tmp')
        shutil.rmtree(savepath, ignore_errors=True)

        model_path = train_test_model_unet_2d_1channel_3classes()
        print('saved unet_2d as {}'.format(model_path))

        model_path = train_test_model_unet_2d_1channel_2classes()
        print('saved unet_2d as {}'.format(model_path))

        model_path = train_test_model_unet_2d_3channels_2classes()
        print('saved unet_2d as {}'.format(model_path))

    def test_is_model_unet_2d(self):

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels/pixels_1.tiff'))

        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model'))

        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_3classes.h5')
        print('model_path: {}'.format(model_path))
        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)
        assert exp._is_model_unet_2d()

        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_2classes.h5')
        print('model_path: {}'.format(model_path))
        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)
        assert exp._is_model_unet_2d()

        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_3channels_2classes.h5')
        print('model_path: {}'.format(model_path))
        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels_rgb/pixels_1.tif'))
        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)
        assert exp._is_model_unet_2d()

    def test_reshape_unet_2d_1channel_3classes(self):

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels/pixels_1.tiff'))

        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model'))

        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_3classes.h5')

        print('model_path: {}'.format(model_path))

        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)

        exp._reshape_unet_2d(size='small')
        assert exp.model_reshaped.input_shape == (None, 112, 112, 1)
        assert exp.model_reshaped.output_shape == (None, 112, 112, 3)

    @pytest.mark.slow
    def test_reshape_unet_2d_1channel_2classes(self):

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels/pixels_1.tiff'))

        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model'))

        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_2classes.h5')

        print('model_path: {}'.format(model_path))

        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)

        exp._reshape_unet_2d(size='small')
        assert exp.model_reshaped.input_shape == (None, 112, 112, 1)
        assert exp.model_reshaped.output_shape == (None, 112, 112, 2)

    @pytest.mark.slow
    def test_reshape_unet_2d_3channels_2classes(self):

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels_rgb/pixels_1.tif'))

        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model'))

        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_3channels_2classes.h5')

        print('model_path: {}'.format(model_path))

        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)

        exp._reshape_unet_2d(size='small')
        assert exp.model_reshaped.input_shape == (None, 112, 112, 3)
        assert exp.model_reshaped.output_shape == (None, 112, 112, 2)

    @pytest.mark.slow
    def test_export_as_tensorflow_model_1(self):
        # 1 channel 3 classes

        tf_version = [int(num) for num in tf.__version__.split('.')]

        if tf_version[0] != 1:
            # deepimagej supports only tensorflow version 1
            return

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels/pixels_1.tiff'))
        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model_1channel_3classes'))
        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_3classes.h5')
        print('model_path: {}'.format(model_path))
        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)
        exp._reshape_unet_2d(size='small')
        exp._export_as_tensorflow_model()

    def test_export_as_tensorflow_model_2(self):
        # 1 channel 2 classes

        tf_version = [int(num) for num in tf.__version__.split('.')]

        if tf_version[0] != 1:
            # deepimagej supports only tensorflow version 1
            return

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels/pixels_1.tiff'))
        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model_1channel_2classes'))
        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_2classes.h5')
        print('model_path: {}'.format(model_path))
        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)
        exp._reshape_unet_2d(size='small')
        exp._export_as_tensorflow_model()

    @pytest.mark.slow
    def test_export_as_tensorflow_model_3(self):
        # 3 channels 2 classes

        tf_version = [int(num) for num in tf.__version__.split('.')]

        if tf_version[0] != 1:
            # deepimagej supports only tensorflow version 1
            return

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels_rgb/pixels_1.tif'))
        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model_3channels_2classes'))
        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_3channels_2classes.h5')
        print('model_path: {}'.format(model_path))
        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)
        exp._reshape_unet_2d(size='small')
        exp._export_as_tensorflow_model()

    def test_update_metadata(self):

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels/pixels_1.tiff'))
        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model_1channel_2classes'))
        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_2classes.h5')
        print('model_path: {}'.format(model_path))
        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)
        exp._reshape_unet_2d(size='small')

        exp._update_metadata(author='Some Name')
        print(exp.metadata)
        assert exp.metadata['input_tensor_dimensions'] == (-1, 112, 112, 1)
        assert exp.metadata['patch_size'] == 112
        assert exp.metadata['author'] == 'Some Name'

    def test_format_xml(self):

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels/pixels_1.tiff'))
        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/exported_model_cpl'))
        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_2classes.h5')
        print('model_path: {}'.format(model_path))
        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)
        exp._reshape_unet_2d(size='small')

        exp._update_metadata(author='Some Name')

        os.makedirs(save_path, exist_ok=True)
        exp.format_xml()

        # xml_path = os.path.join(
        #     base_path,
        #     '../../templates/deepimagej101/config.xml')
        #
        # xml_save_path = os.path.join(
        #     base_path,
        #     '../test_data/tmp/tst.xml')
        #
        # tree = ET.parse(xml_path)
        #
        # root = tree.getroot()
        #
        # for child in root:
        #     print(child.tag, child.attrib)
        #
        # ps = tree.find('ModelCharacteristics').find('Padding')
        # ps.text = '512'
        # print('ps')
        # print(ps.text)
        #
        # tree.write(xml_save_path)
        # #print(ET.tostring(tree))


        assert False
