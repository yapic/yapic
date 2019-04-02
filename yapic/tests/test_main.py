# import pyilastik

from unittest import TestCase
from yapic_io.training_batch import TrainingBatch
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.dataset import Dataset
from yapic.training_project import TrainingProject
import yapic.network as netw
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf


tf.logging.set_verbosity(5)
base_path = os.path.dirname(__file__)

class TestTrainingProject(TestCase):

    def test_tmp(self):



        img_path = os.path.join(
            base_path,
            '../test_data/shapes/pixels/*')
        label_path = os.path.join(
            base_path,
            '../test_data/shapes/labels.ilp')
        c = IlastikConnector(img_path, label_path)
        d = Dataset(c)


        t = TrainingProject(d)
        t.make_model('unet_2d', (1, 572, 572))

        print('pixels')
        print(t.data.pixels().shape)
        print('weights')
        print(t.data.weights().shape)

        t.train(max_epochs=3)
        print(t)


        assert False





class TestNetwork(TestCase):

    def test_setup_network(self):

        n_classes = 2
        size_czxy = (3, 1, 572, 572)

        model = netw.setup_network('unet_2d', n_classes, size_czxy)

        self.assertEqual(model.input_shape, (None, 3, 1, 572, 572))

    def test_compile_model(self):

        n_classes = 2
        size_czxy = (3, 1, 572, 572)

        model = netw.setup_network('unet_2d', n_classes, size_czxy)

        model = netw.compile_model(model)

    def test_make_model(self):

        n_classes = 2
        size_czxy = (3, 1, 572, 572)

        model = netw.make_model('unet_2d', n_classes, size_czxy)
