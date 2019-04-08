# import pyilastik

from unittest import TestCase
from yapic_io.training_batch import TrainingBatch
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.dataset import Dataset
from yapic.session import Session
import yapic.network as netw
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import tensorflow as tf
import keras
import numpy as np
from keras import backend as K

# tf.logging.set_verbosity(0)
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


    def test_loss_func(self):
        # [3.295837  1.3862944]
        y_true = np.array([[0., 0., 0., 0., 0],
                           [0., 0., 1., 0., 1],
                           [0., 0., 1., 0., 1]])

        y_pred = np.array([[1., 1., 0., 1., 0],
                           [0., 0., 1., 0., 1],
                           [0., 0., 1., 0., 1]])


        y_true = K.variable(y_true)
        y_pred = K.variable(y_pred)

        loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        corr = netw.correct_mean(y_true)
        loss_corr = loss * netw.correct_mean(y_true)
        loss = K.get_value(loss)
        loss_corr = K.get_value(loss_corr)
        corr = K.get_value(corr)
        print(loss)
        print(loss_corr)
        print(corr)
        nr_labels = K.get_value(netw.count_labels(y_true))
        print('nr labels: {}'.format(nr_labels))
        assert False
