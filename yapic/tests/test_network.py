from unittest import TestCase
import yapic.network as netw


class TestNetwork(TestCase):

    def test_setup_unet_2d(self):

        n_classes = 2
        size_czxy = (3, 1, 572, 572)

        model = netw.setup_network('unet_2d', n_classes, size_czxy)

        self.assertEqual(model.input_shape, (None, 1, 572, 572, 3))
        self.assertEqual(model.output_shape, (None, 1, 358, 358, 2))
