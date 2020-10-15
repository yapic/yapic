import logging
import re

logger = logging.getLogger(__name__)


def handle_augmentation_setting(mbatch, augment):

    mbatch.augment_by_flipping(False)
    mbatch.augment_by_rotation(False)
    mbatch.augment_by_shear(False)

    augment = set(augment.split('+'))

    if 'flip' in augment:
        mbatch.augment_by_flipping(True)
        logger.info('activate augmentation mode: flip')

    if 'rotate' in augment:
        mbatch.augment_by_rotation(True)
        logger.info('activate augmentation mode: rotate')

    if 'shear' in augment:
        mbatch.augment_by_shear(True)
        logger.info('activate augmentation mode: shear')

    if not mbatch.augmentation:
        logger.info('Augmentation is not active')

    undefined = augment - {'flip', 'rotate', 'shear'}
    if undefined:
        raise ValueError(
            'Incorrect augmentation mode(s): {}'.format(undefined))


def handle_normalization_setting(mbatch, normalize_str):
    if normalize_str in ['local', 'local_z_score', 'off']:
        mbatch.set_normalize_mode(normalize_str)
        logger.info('set normalization mode to {}'.format(normalize_str))
        return

    pattern = r'(global)_(-?\d{1,10})\+(-?\d{1,10})'
    regex_match = re.match(pattern, normalize_str)

    assert regex_match, 'Incorrect normalize mode: {}'.format(normalize_str)

    mode, min, max = regex_match.groups()
    minmax = (int(min), int(max))
    assert minmax[0] < minmax[1], 'max larger than min: {}'.format(minmax)
    mbatch.set_normalize_mode(mode, minmax=minmax)
