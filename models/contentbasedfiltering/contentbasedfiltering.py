import datetime
import sys
import time
from models.recommender import Recommender
from support.cosinesimilarity import CosineSimilarity
from support.utility import check_matrix
import scipy.sparse as sp
import numpy as np
from sklearn.linear_model import ElasticNet

#######################################################################
############# CONTENT-BASED FILTERING RECOMMENDER SYSTEM ##############
#######################################################################

class CBFRecommender(Recommender):

    """ SLIM RMSE RECOMMENDATION SYSTEM ALGORITHM
        slim implementation that minimizes the Root Mean Squared Error (RMSE)
        using the ElasticNet class provided by the Sklearn module.
        there is an option to add side information to the algorithm """

    N_CONFIG = 0

    def __init__(self, train, test, validation, targets, albums=None, artists=None, duration=None,
                 log_filename="cbf_config.txt"):
        super(CBFRecommender, self).__init__(train, test, validation, targets, log_filename)
        self.ICM_albums = albums
        self.ICM_artists = artists
        self.ICM_duration = duration
        self.id_album, self.id_artist, self.id_duration = 0, 1, 2
        self.configuration_txt = "CONTENT-BASED FILTERING RECOMMENDER SYSTEM"

    def fit(self, topks=(70, 70, 70), similarities=('cosine', 'cosine', 'cosine'), shrinks=(100, 100, 100),
            weights=(2.0, 0.8, 0.0), verbose=True):

        """ Fits content-based filtering model
            params: tuples that represents the value for each ICM
                    (album, artist, duration) """

        self.similarities = similarities
        self.topks = topks
        self.shrinks = shrinks
        self.weights = weights

        start_time = time.time()

        # compute similarity for ICM_albums
        sim_obj = CosineSimilarity(dataMatrix=self.ICM_albums.T, similarity=similarities[self.id_album],
                                   topK=topks[self.id_album], shrink=shrinks[self.id_album])
        W_albums = sim_obj.compute_similarity()

        # compute similarity for ICM_artists
        sim_obj = CosineSimilarity(dataMatrix=self.ICM_artists.T, similarity=similarities[self.id_artist],
                                   topK=topks[self.id_artist], shrink=shrinks[self.id_artist])
        W_artists = sim_obj.compute_similarity()

        # compute similarity for ICM_duration
        sim_obj = CosineSimilarity(dataMatrix=self.ICM_duration.T, similarity=similarities[self.id_duration],
                                   topK=topks[self.id_duration], shrink=shrinks[self.id_duration])
        W_duration = sim_obj.compute_similarity()

        self.W = W_albums.multiply(weights[self.id_album]) + W_artists.multiply(weights[self.id_artist]) \
                 + W_duration.multiply(weights[self.id_duration])

        self.predicted_URM = self.URM_train.dot(self.W)

        self.set_configuration()

        if verbose:
            print('CONTENT-BASED FILTERING computed in {:.2f} minutes'.format((time.time() - start_time) / 60))


    def set_configuration(self):

        """ Sets the configuration of the current algorithm """

        now = datetime.datetime.now()
        self.configuration_txt = "CONTENT-BASED FILTERING configuration {}:\n" \
                                 "date    ==>  {}\n" \
                                 "model legend : albums-artists-duration\n" \
                                 "model   ==>  similarities={}; topks={}; shrinks={}; " \
                                 "weights={}\n".format(CBFRecommender.N_CONFIG, now,
                                                       self.similarities, self.topks,
                                                       self.shrinks, self.weights)
        # increment the number of configuration
        CBFRecommender.N_CONFIG += 1

