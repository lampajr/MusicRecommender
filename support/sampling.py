import numpy as np

###################################################################
#################### BPR SAMPLING CLASS ###########################
###################################################################


class BPRSampling(object):

    """ BPR Sampling class """

    def __init__(self, data):
        super(BPRSampling, self).__init__()

        self.NUM_MIN_INTERACTIONS = 0

        # data matrix from which sampling
        self.URM = data
        self.n_playlists, self.n_tracks = data.shape

    def __sample_playlist(self):

        """ Samples a playlist that has at least one interaction """

        while True:
            p_id = np.random.randint(low=0, high=self.n_playlists)
            num_interactions = self.URM[p_id].nnz
            if num_interactions > self.NUM_MIN_INTERACTIONS:
                return p_id

    def __sample_tracks(self, p_id):

        """
        Samples the positive track and the negative one for a given playlist id
        :param p_id: playlist_id
        :return: tuple of two tracks (positive, negative)
        """
        positive_tracks = self.URM[p_id].indices
        pos_track = np.random.choice(a=positive_tracks)

        while True:
            neg_track = np.random.randint(low=0, high=self.n_tracks)
            if neg_track not in positive_tracks:
                return pos_track, neg_track

    def generate_sample(self):

        """ Sample a triple : playlist, positive track, negative track """

        p_id = self.__sample_playlist()
        pos_track, neg_track = self.__sample_tracks(p_id=p_id)

        return p_id, pos_track, neg_track


