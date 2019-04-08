from unittest import TestCase
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.dataset import Dataset
from yapic.session import Session
import os
import skimage
import numpy as np

base_path = os.path.dirname(__file__)


class TestEnd2End(TestCase):

    def test_shape_data(self):

        # train a classifier and predict training data

        os.environ['CUDA_VISIBLE_DEVICES']='2';

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

        c = IlastikConnector(img_path,
                             label_path,
                             savepath=os.path.abspath(savepath))
        d = Dataset(c)
        t = Session(d)

        t.make_model('unet_2d', (1, 572, 572))

        t.train(max_epochs=10,
                steps_per_epoch=24,
                log_filename=os.path.join(savepath, 'log.csv'))
        t.predict()



        # read prediction images and compare with validation data

        def read_images(image_nr, class_nr):
            if class_nr == 1:
                shape = 'circles'
            if class_nr == 2:
                shape = 'triangles'

            filename = os.path.join(
                                savepath,
                                'pixels_{}_class_{}.tif'.format(image_nr,
                                                                class_nr))
            print(filename)
            prediction_img = np.squeeze(skimage.io.imread(filename))
            filename = os.path.join(
                                    savepath,
                                    '../shapes/val/{}_{}.tiff'.format(
                                        shape,
                                        image_nr))
            print(filename)
            val_img = np.squeeze(skimage.io.imread(filename))
            return prediction_img, val_img

        prediction_img, val_img = read_images(1, 1)
        accuracy = np.mean(prediction_img[val_img>0][:])
        self.assertTrue(accuracy > 0.9)

        prediction_img, val_img = read_images(1, 2)
        accuracy = np.mean(prediction_img[val_img>0][:])
        self.assertTrue(accuracy > 0.9)

        prediction_img, val_img = read_images(2, 1)
        accuracy = np.mean(prediction_img[val_img>0][:])
        self.assertTrue(accuracy > 0.9)

        prediction_img, val_img = read_images(2, 2)
        accuracy = np.mean(prediction_img[val_img>0][:])
        self.assertTrue(accuracy > 0.9)
