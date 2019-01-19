import numpy as np
from .utility import read_data, create_sparse


########################################
############# DATASET ##################
########################################

class Dataset(object):

    """ DATASET class """

    def __init__(self, split_traintestvalidation=(0.8, 0.2, 0.0), verbose=True):
        super(Dataset, self).__init__()
        self.verbose = verbose
        self.split_traintestvalidation = split_traintestvalidation
        self.URM_train = None
        self.URM_test = None
        self.URM_validation = None
        self.target_playlists = None
        self.ICM_albums = None
        self.ICM_artists = None
        self.ICM_duration = None

    def load_data(self):

        # load the whole dataset
        self._load_URM()
        self._load_ICM()
        self._load_targets()

    def _load_URM(self):

        # load the train data
        train = read_data(filename='train.csv')

        # create the mask for splitting the train set in 3 subsets

        split = np.random.choice([1,2,3], size=len(train), p=self.split_traintestvalidation)

        mask = split == 1
        self.URM_train = create_sparse(data=train[mask], row='playlist_id', col='track_id')

        if self.verbose:
            print('Training set generated!')

        if self.split_traintestvalidation[1] != 0.0:
            mask = split == 2
            self.URM_test = create_sparse(data=train[mask], row='playlist_id', col='track_id')

            if self.verbose:
                print('Test set generated!')

        if self.split_traintestvalidation[2] != 0.0:
            mask = split == 3
            self.URM_validation = create_sparse(data=train[mask], row='playlist_id', col='track_id')

            if self.verbose:
                print('Validation set generated!')

    def _load_ICM(self):

        # load dataset
        tracks = read_data(filename='tracks.csv')

        # generate the sparse matrices
        self.ICM_albums = create_sparse(data=tracks, row='track_id', col='album_id')
        self.ICM_artists = create_sparse(data=tracks, row='track_id', col='artist_id')
        self.ICM_duration = create_sparse(data=tracks, row='track_id', col='duration_sec')

    def _load_targets(self):

        # load the target playlists set
        target_playlist = read_data(filename='target_playlists.csv')

        # generate the np.array of the targets
        self.target_playlists = np.array(target_playlist['playlist_id'])



