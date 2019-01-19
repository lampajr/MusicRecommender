import time
import numpy as np

from models.recommender import Recommender
from lightfm import LightFM


###################################################################
############# PURE LIGHTFM COLLABORATIVE FILTERING ################
###################################################################


class LightFMRecommender(Recommender):

    """ PURE LIGHTFM COLLABORATIVE FILTERING """

    N_CONFIG = 0

    def __init__(self, train, test, validation, targets, log_filename='lightfmcf_config.txt'):
        super(LightFMRecommender, self).__init__(train, test, validation, targets, log_filename)
        self.configuration_txt = "PURE LIGHTFM COLLABORATIVE FILTERING"

    def fit(self, item_alpha=1e-5, user_alpha=1e-4, learning_schedule='adadelta', num_components=250, epochs=30, threads=2):
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.learning_schedule = learning_schedule
        self.num_components = num_components
        self.epochs = epochs
        self.threads = threads

    def train(self):
        start_time = time.time()

        # Let's fit a WARP model: these generally have the best performance.
        self.model = LightFM(loss='warp',
                             item_alpha=self.item_alpha,
                             user_alpha=self.user_alpha,
                             learning_schedule=self.learning_schedule,
                             no_components=self.num_components)

        # Run 3 epochs and time it.
        self.model = self.model.fit(self.URM_train, epochs=self.epochs, num_threads=self.threads)
        print("LightFM training model fitted in {:.2f} seconds".format(time.time() - start_time))

    def compute_predicted_ratings(self, playlist_id):
        return self.model.predict(user_ids=playlist_id, item_ids=np.arange(self.n_tracks), item_features=None,
                                  user_features=None, num_threads=self.threads)
