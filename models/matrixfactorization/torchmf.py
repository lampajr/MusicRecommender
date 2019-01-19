import datetime
import time
from models.recommender import MatrixFactorizationRecommender
import numpy as np
import pytorch


#########################################################################################
############# PYTORCH MATRIX FACTORIZATION RECOMMENDER SYSTEM ALGORITHM #################
#########################################################################################

class TorchMFRecommender(MatrixFactorizationRecommender):

    """ PYTORCH MATRIX FACTORIZATION RECOMMENDER SYSTEM ALGORITHM """

    N_CONFIG = 0

    def __init__(self, train, test, validation, targets, log_filename='torchmf_config.txt'):
        super(TorchMFRecommender, self).__init__(train, test, validation, targets, log_filename)

    def fit(self, epochs=20, latent_factors=10, learning_rate=0.001):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.latent_factors = latent_factors

        # define the device used by pytorch

    def train(self):
        pass


class TorchModel():
    pass
