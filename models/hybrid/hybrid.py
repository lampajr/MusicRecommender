import datetime

from models.collaborativefiltering.lightfmcf import LightFMRecommender
from models.recommender import Recommender
from models.collaborativefiltering.itembasedcf import ItemBasedCFRecommender
from models.collaborativefiltering.userbasedcf import UserBasedCFRecommender
from models.contentbasedfiltering.contentbasedfiltering import CBFRecommender
from models.matrixfactorization.alsmf import ALSMFRecommender
from models.slim.slimbpr import SLIMBPRRecommender
from models.slim.sslimrmse import SSLIMRMSERecommender
from support.utility import get_tops
import numpy as np


#############################################################################
##################### HYBRID RECOMMENDER SYSTEM #############################
#############################################################################


class HybridRecommender(Recommender):

    """ HYBRID RECOMMENDER SYSTEM """

    N_CONFIG = 0

    def __init__(self, train, test, validation, targets, albums, artists, duration,
                 subfolder="../", log_filename='hybrid_config.txt'):
        super(HybridRecommender, self).__init__(train, test, validation, targets, subfolder, log_filename)
        self.ICM_albums = albums
        self.ICM_artists = artists
        self.ICM_duration = duration

        self.configuration_txt = "HYBRID RECOMMENDER SYSTEM"

        self.k = None

    def fit(self, k=None, light=False, item=True, user=True, content=True, rmse=True, bpr=True, als=True):

        """ Fits all the models """

        self.set_k(k=k)  # number of tracks to keep during algorithms merging

        if light:
            ### LIGHTFM CF ###
            self.LFMCF = LightFMRecommender(self.URM_train, self.URM_test,
                                            self.URM_validation, self.target_playlists, subfolder=self.subfolder)
            self.LFMCF.fit()
            self.LFMCF.train()


        if item:
            ### Item-based collaborative filtering ###
            self.ItemBased = ItemBasedCFRecommender(self.URM_train, self.URM_test,
                                                    self.URM_validation, self.target_playlists,
                                                    subfolder=self.subfolder)
            self.ItemBased.fit()


        if user:
            ### User-based collaborative filtering ###
            self.UserBased = UserBasedCFRecommender(self.URM_train, self.URM_test,
                                                    self.URM_validation, self.target_playlists,
                                                    subfolder=self.subfolder)
            self.UserBased.fit()


        if content:
            ### Content-based filtering ###
            self.ContentBased = CBFRecommender(self.URM_train, self.URM_test, self.URM_validation,
                                               self.target_playlists, self.ICM_albums, self.ICM_artists,
                                               self.ICM_duration, subfolder=self.subfolder)
            self.ContentBased.fit()


        if als:
            ### ALS Matrix Factorization ###
            self.ALS = ALSMFRecommender(self.URM_train, self.URM_test,
                                        self.URM_validation, self.target_playlists, subfolder=None)
            self.ALS.fit()
            self.ALS.train()


        if bpr:
            ### SLIM BPR SGD ###
            self.SLIMBPR = SLIMBPRRecommender(self.URM_train, self.URM_test,
                                              self.URM_validation, self.target_playlists,
                                              subfolder=self.subfolder)
            self.SLIMBPR.fit()
            self.SLIMBPR.train()


        if rmse:
            ### SSLIM RMSE ###
            self.SSLIM = SSLIMRMSERecommender(self.URM_train, self.URM_test, self.URM_validation,
                                              self.target_playlists, self.ICM_albums, self.ICM_artists,
                                              self.ICM_duration, subfolder=self.subfolder)
            self.SSLIM.fit()
            self.SSLIM.add_side_information()
            self.SSLIM.train()



    def set_k(self, k):

        """ Set the k value """

        if k is None:
            self.k = self.n_tracks
        else:
            self.k = k

        #self.set_configuration()

    def set_weights(self, IBCFweight=None, UBCFweight=0.83, LFMCFweight=None, CBFweight=0.83, SSLIMweight=1.0,
                    ALSweight=0.345, SLIMBPRweight=0.015):

        """ Sets the weights for every algorithm involved in the hybrid recommender """

        self.IBCFweight = IBCFweight
        self.UBCFweight = UBCFweight
        self.LFMCFweight = LFMCFweight
        self.CBFweight = CBFweight
        self.SSLIMweight = SSLIMweight
        self.ALSweight = ALSweight
        self.SLIMBPRweight = SLIMBPRweight

        self.set_configuration()

    def compute_predicted_ratings(self, playlist_id):

        """ Computes predicted ratings across all different recommender algorithms """

        playlist_profile = self.URM_train[playlist_id]
        predicted_ratings = np.zeros(shape=self.n_tracks, dtype=np.float32)

        if self.IBCFweight is not None:
            itemcf_exp_ratings = self.ItemBased.predicted_URM[playlist_id].toarray().ravel()
            tops = get_tops(itemcf_exp_ratings, self.k)
            predicted_ratings[tops] += np.multiply(itemcf_exp_ratings[tops], self.IBCFweight)

        if self.UBCFweight is not None:
            usercf_exp_ratings = self.UserBased.predicted_URM[playlist_id].toarray().ravel()
            tops = get_tops(usercf_exp_ratings, self.k)
            predicted_ratings[tops] += np.multiply(usercf_exp_ratings[tops], self.UBCFweight)

        if self.LFMCFweight is not None:
            lightfm_exp_ratings = self.LFMCF.compute_predicted_ratings(playlist_id=playlist_id)
            tops = get_tops(lightfm_exp_ratings, self.k)
            predicted_ratings[tops] += np.multiply(lightfm_exp_ratings[tops], self.LFMCFweight)

        if self.CBFweight is not None:
            cbf_exp_ratings = self.ContentBased.predicted_URM[playlist_id].toarray().ravel()
            tops = get_tops(cbf_exp_ratings, self.k)
            predicted_ratings[tops] += np.multiply(cbf_exp_ratings[tops], self.CBFweight)

        if self.ALSweight is not None:
            mfals_exp_ratings = np.dot(self.ALS.X[playlist_id], self.ALS.Y.T)
            tops = get_tops(mfals_exp_ratings, self.k)
            predicted_ratings[tops] += np.multiply(mfals_exp_ratings[tops], self.ALSweight)

        if self.SSLIMweight is not None:
            slimrmse_exp_ratings = self.SSLIM.predicted_URM[playlist_id].toarray().ravel()
            tops = get_tops(slimrmse_exp_ratings, self.k)
            predicted_ratings[tops] += np.multiply(slimrmse_exp_ratings[tops], self.SSLIMweight)

        if self.SLIMBPRweight is not None:
            slimbpr_exp_ratings = playlist_profile.dot(self.SLIMBPR.W).toarray().ravel()
            tops = get_tops(slimbpr_exp_ratings, self.k)
            predicted_ratings[tops] += np.multiply(slimbpr_exp_ratings[tops], self.SLIMBPRweight)

        return predicted_ratings

    def set_configuration(self):

        """ Sets the configuration of the current algorithm """

        now = datetime.datetime.now()
        self.configuration_txt = "HYBRID RECOMMENDER configuration {}:\n" \
                                 "date  ==>  {}\n" \
                                 "model :" \
                                 "- k = {}, " \
                                 "- item-based CF ==> {}, - user-based CF ==> {}, " \
                                 "- cbf ==> {}, - slim-rmse ==> {}, - lfmcf ==> {}" \
                                 "- als-mf ==> {}, - slim-bpr ==> {}\n".format(HybridRecommender.N_CONFIG, now, self.k,
                                                                               self.IBCFweight, self.UBCFweight,
                                                                               self.CBFweight, self.SSLIMweight,
                                                                               self.LFMCFweight, self.ALSweight,
                                                                               self.SLIMBPRweight)

        # increment the number of configuration
        HybridRecommender.N_CONFIG += 1


