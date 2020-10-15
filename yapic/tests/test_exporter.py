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
    model_filename = os.path.join(savepath,
                                  'model_unet_2d_1channel_3classes.h5')
    if os.path.isfile(model_filename):
        return model_filename

    os.makedirs(savepath, exist_ok=True)

    t = Session()
    t.load_training_data(img_path, label_path)
    t.make_model('unet_2d', (1, 220, 220))
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
    model_filename = os.path.join(savepath,
                                  'model_unet_2d_1channel_2classes.h5')
    if os.path.isfile(model_filename):
        return model_filename

    os.makedirs(savepath, exist_ok=True)

    t = Session()
    t.load_training_data(img_path, label_path)
    t.make_model('unet_2d', (1, 220, 220))
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
    model_filename = os.path.join(savepath,
                                  'model_unet_2d_3channels_2classes.h5')
    if os.path.isfile(model_filename):
        return model_filename

    os.makedirs(savepath, exist_ok=True)

    t = Session()
    t.load_training_data(img_path, label_path)
    t.make_model('unet_2d', (1, 220, 220))
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
    model_filename = os.path.join(savepath, 'model_convnet.h5')
    if os.path.isfile(model_filename):
        return model_filename

    os.makedirs(savepath, exist_ok=True)

    t = Session()
    t.load_training_data(img_path, label_path)
    t.make_model('convnet_for_unittest', (1, 100, 100))
    t.train(max_epochs=2,
            steps_per_epoch=2,
            log_filename=os.path.join(savepath, 'log.csv'),
            model_filename=model_filename)

    return model_filename


class TestDeepimagejExporter(TestCase):

    def setUp(self):

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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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
        assert exp.metadata['name'] == 'exported_model_1channel_2classes'
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

        exp._update_metadata(author='Some Name',
                             version='1.0.0',
                             url='https://my-site.org',
                             credit='some other names',
                             reference='Name et al. 2020'
                             )

        os.makedirs(save_path, exist_ok=True)
        exp._format_xml()

    @pytest.mark.slow
    def test_export_as_deepimagej(self):
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
            '../test_data/tmp/my_packaged_u_net'))
        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_2classes.h5')
        print('model_path: {}'.format(model_path))

        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)

        shutil.rmtree(save_path, ignore_errors=True)
        exp.export_as_deepimagej(
            author='Some Name',
            version='1.0.0',
            url='https://my-site.org',
            credit='some other names',
            reference='Name et al. 2020',
            size='small')

    @pytest.mark.slow
    def test_apply_model(self):

        example_image_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/shapes/pixels_small/pixels_1.tif'))
        save_path = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/my_packaged_u_net'))
        temp_dir = os.path.abspath(os.path.join(
            base_path,
            '../test_data/tmp/temp_dir'))
        model_path = os.path.join(
            base_path,
            '../test_data/tmp/model_unet_2d_1channel_2classes.h5')
        print('model_path: {}'.format(model_path))
        exp = DeepimagejExporter(model_path,
                                 save_path,
                                 example_image_path)

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        exp.apply_model(normalization_mode='local')
