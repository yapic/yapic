"""Yet Another Pixel Classifier.

Usage:
  yapic train <network> <image_path> <label_path> [options]
  yapic predict <network> <image_path> <output_path> [options]

  <network>               Either a model file in h5 format to use a pretrained
                          model or specific string to initialize a new model.
                          Choose 'unet_2d' or 'unet_multi_z' to initialize a
                          new model. Use path/to/my/pretrained_model.h5 to
                          continue training of a pretrained keras model.
  <image_path>            Path to image files. You can use wildcards, e.g.
                          "my_data/*.tif".
  <label_path>            Path to label files. Either tif or ilp (Ilastik project file).
                          Wildcards are supported.
                          Examples: "path/to/my_ilastik_project.ilp",
                                    "path/to/my_label_images/*.tif"

Options:
  -n --normalize=NORM     Set pixel normalization scope [default: local]
                          For minibatch-wise normalization choose 'local_z_score' or 'local'.
                          For global normalization use global_<min>+<max>
                          (e.g. 'global_0+255' for 8-bit images and 'global_0+65535'
                          for 16-bit images)
                          Choose 'off' to deactivate.
  --cpu                   Train using the CPU (not recommended).
  --gpu=VISIBLE_DEVICES   If you wanrt to use specific gpus. To use gpu 0,
                          set '0'. To use gpus 2 and 3, set '2,3'
  -h --help               Show this screen.
  --version               Show version.

Train Options:
  -e --epochs=MAX_EPOCHS  Maximum number of epochs to train [default: 5000].
  -a --augment=AUGMENT    Set augmentation method for training [default: flip]
                          Choose 'flip' and/or 'rotate' and/or 'shear'.
                          Use '+' to specify multiple augmentations (e.g. flip+rotate).
  -v --valfraction=VAL    Fraction of images to be used for validation [default: 0.2].
  -f --file=CLASSIFER     Path to trained model [default: model.h5].
  --steps=STEPS           Steps per epoch [default: 50].
  --equalize              Equalize label weights to promote less frequent labels.
  --csvfile=LOSSDATA      Path to csv file for training loss data [default: loss.csv].


"""
from docopt import docopt

import os
import sys
from yapic.session import Session

import logging
logger = logging.getLogger(__name__)


def main(args):

    s = Session()

    if args['--cpu']:
        # deactivate gpu for tensorflow
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args['--gpu']:
        # define gpu hardware
        os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']

    image_path = os.path.abspath(os.path.expanduser(args['<image_path>']))
    model_name = args['<network>']
    norm_string = args['--normalize']


    if args['train']:

        label_path = os.path.abspath(os.path.expanduser(args['<label_path>']))
        aug_string = args['--augment']
        max_epochs = int(args['--epochs'])
        steps_per_epoch = int(args['--steps'])
        log_filename = args['--csvfile']
        model_export_filename = args['--file']
        valfraction = float(args['--valfraction'])

        s.load_training_data(image_path, label_path)

        models_available = ['unet_2d',
                            'unet_multi_z',
                            'convnet_for_unittest']

        if os.path.isfile(model_name):
            s.load_model(model_name)
        elif model_name in models_available:
            size_xy = 572
            if model_name == 'unet_2d' or model_name == 'convnet_for_unittest':
                size_z = 1
            if model_name == 'unet_multi_z':
                size_z = 5
            if model_name == 'convnet_for_unittest':
                size_xy = 100

            s.make_model(model_name, (size_z, size_xy, size_xy))

        s.set_normalization(norm_string)
        s.set_augmentation(aug_string)

        if valfraction > 0:
            s.define_validation_data(valfraction)

        s.train(max_epochs=max_epochs,
                steps_per_epoch=steps_per_epoch,
                log_filename=log_filename,
                model_filename=model_export_filename)

    if args['predict']:
        output_path = args['<output_path>']
        assert os.path.isfile(model_name), '<network> must be a h5 model file'
        s.load_prediction_data(image_path, output_path)
        s.load_model(model_name)
        s.set_normalization(norm_string)
        s.predict()


def entry_point():
    arguments = docopt(__doc__, version='YAPiC 0.1.0')
    res = main(arguments)
    sys.exit(res)


if __name__ == '__main__':
    entry_point()
