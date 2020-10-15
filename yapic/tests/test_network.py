from unittest import TestCase
import yapic.network as netw
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''


class TestNetwork(TestCase):

    def test_setup_unet_2d(self):

        n_classes = 2
        size_czxy = (3, 1, 572, 572)

        model = netw.setup_network('unet_2d', n_classes, size_czxy)

        self.assertEqual(model.input_shape, (None, 1, 572, 572, 3))
        self.assertEqual(model.output_shape, (None, 1, 358, 358, 2))

    def test_setup_unet_multi_z(self):

        n_classes = 2
        size_czxy = (3, 5, 572, 572)

        model = netw.setup_network('unet_multi_z', n_classes, size_czxy)

        self.assertEqual(model.input_shape, (None, 5, 572, 572, 3))
        self.assertEqual(model.output_shape, (None, 1, 354, 354, 2))

    def test_setup_convnet_for_unittest(self):

        n_classes = 2
        size_czxy = (3, 1, 150, 150)

        model = netw.setup_network('convnet_for_unittest',
                                   n_classes,
                                   size_czxy)

        self.assertEqual(model.input_shape, (None, 1, 150, 150, 3))
        self.assertEqual(model.output_shape, (None, 1, 138, 138, 2))
