
class TrainingProject(object):
    '''
    Provides connectors to pixel data source and assigned weights for
    classifier training.
    Provides methods for getting image tiles and data augmentation for
    classifier training, as well as writing classifier output tile-by-tile
    to target images.



    Parameters
    ----------
    data : yapic_io.TrainingBatch
        Connector object for binding pixel and label data

    Notes
    -----
    Pixel data is loaded lazily to allow images of arbitrary size.
    Pixel data is cached in memory for repeated requests.
    '''

    def __init__(data, valfraction=0):

        self.data = data
        if valfraction > 0:
            self.data_val = self.data.split(valfraction)
        else:
            self.data_val = data
