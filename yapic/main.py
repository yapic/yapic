"""Yet Another Pixel Classifier.

Usage:
  yapic.py train <network> <image_path> <label_path> [-c] [options]
  yapic.py predict <network> <image_path> <output_path> [--file=CLASSIFIER] [options]

  <network>               Either a model file in h5 format to use a pretrained
                          model or specific string to initialize a new model.
  <image_path>            Path to image files (by default *.tif is appended).
  <label_path>            Path to label files. Either tif or ilp (Ilastik project file).
                          Wildcards are supported for tif. By default *.tif is appended.

Options:
  -n --normalize=NORM     Set pixel normalization scope [default: local]
                          For minibatch-wise normalization choose 'local_z_score' or 'local'.
                          For global normalization use global_<min>+<max>
                          (e.g. 'global_0+255' for 8-bit images and 'global_0+65535'
                          for 16-bit images)
                          Choose 'off' to deactivate.
  -h --help               Show this screen.
  --version               Show version.

Train Options:
  -e --epochs=MAX_EPOCHS  Maximum number of epochs to train [default: 2500].
  -l --learning=RATE      Set the learning rate [default: 1e-3].
  -m --momentum=MOMENTUM  Set the learning rate [default: 0.9].
  -a --augment=AUGMENT    Set augmentation method for training [default: flip]
                          Choose 'flip' and/or 'rotate' and/or 'shear'.
                          Use '+' to specify multiple augmentations (e.g. flip+rotate).
  -v --valfraction=VAL    Fraction of images to be used for validation [default: 0.16].
  --valfrequency=VALFREQ  Run validation step after this number of seconds [default: 30].
  --equalize              Equalize label weights to promote less frequent labels.
  --csvfile=LOSSDATA      Path to csv file for training loss data [default: loss.csv].

Predict Options:
  -f --file=CLASSIFER     Path to classifer [default: model.h5].

"""
from docopt import docopt

import os
import sys
from session import Session

import logging
logger = logging.getLogger(__name__)


def main(args):

    print(args)

    s = Session()

    if args['train']:
        s.load_training_data(args['<image_path>'], args['<label_path>'])
    if args['predict']:
        s.load_prediction_data(args['<image_path>'], args['<output_path>'])





def entry_point():
    arguments = docopt(__doc__, version='YAPiC 0.1.0')
    res = main(arguments)
    sys.exit(res)

if __name__ == '__main__':
    entry_point()
