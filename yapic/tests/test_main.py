from unittest import TestCase
from yapic_io.training_batch import TrainingBatch
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.dataset import Dataset
import os

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

        size = (1, 50, 50)
        pad = (0, 0, 0)

        m = TrainingBatch(d, size, padding_zxy=pad)
