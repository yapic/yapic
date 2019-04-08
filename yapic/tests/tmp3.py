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


label_path = '/mnt/data-fast/christoph/dennis_actin_tem/data/MyProject_190214.ilp'
img_path = '/mnt/data-fast/christoph/dennis_actin_tem/data/190204_EM_fiji2/*.tif'
c = IlastikConnector(img_path, label_path, savepath=os.path.abspath(savepath))
d = Dataset(c)


t = Session(d)
t.make_model('unet_2d_channels_last', (1, 572, 572))
#t.make_model('unet_2d', (1, 256, 256))

next(t.data)
print(t.data.pixels())
next(t.data)



t.train(max_epochs=10000, steps_per_epoch=24, log_filename='log.csv')
t.predict()
t.model.save('my_model.h5')
print(t.history.history)
