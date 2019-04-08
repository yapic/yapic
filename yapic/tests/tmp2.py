from yapic_io.training_batch import TrainingBatch
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.dataset import Dataset
from yapic.session import Session
import os
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

base_path = os.path.dirname(__file__)

img_path = os.path.join(
    base_path,
    '../test_data/shapes/pixels/*')
label_path = os.path.join(
    base_path,
    '../test_data/shapes/labels.ilp')
savepath = os.path.join(
    base_path,
    '../test_data/tmp')


label_path = '/mnt/data-fast/christoph/microglomeruli/data/microglomeruli_recognition_luigi_2.ilp'
img_path = '/mnt/data-fast/christoph/microglomeruli/data/pixels_fiji/*.tif'
c = IlastikConnector(img_path, label_path, savepath=os.path.abspath(savepath))
d = Dataset(c)

data = TrainingBatch(d,
                    (1, 500, 500),
                    padding_zxy=(0, 20, 20))

next(data)
data.pixels()
data.weights()
