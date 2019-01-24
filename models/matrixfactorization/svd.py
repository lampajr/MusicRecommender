import datetime
import time

import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd

from models.recommender import MatrixFactorizationRecommender


class SVDRecommender(MatrixFactorizationRecommender):

    """ SINGULAR VALUE DECOMPOSITION MATRIX FACTORIZATION RECOMMENDER SYSTEM ALGORITHM """

    N_CONFIG = 0

    def __init__(self, train, test, validation, targets, log_filename='svd_config.txt'):
        super(SVDRecommender, self).__init__(train, test, validation, targets, log_filename)

    def fit(self, latent_factors=460, scipy=True):

        """ Fits the SVD MF model """

        self.latent_factors = latent_factors
        self.scipy = scipy

        self.set_configuration()

    def train(self, verbose=True):

        """ Trains the ALS MF model """

        start_time = time.time()

        if verbose:
            print('SVD training started...')

        if self.scipy:
            print('computing u, s, v  using scipy model ...')
            u, s, v = svds(self.URM_train.astype('float'), k=self.latent_factors, which='LM')
        else:
            print('computing u, s, v using sklearn model ...')
            u, s, v = randomized_svd(self.URM_train, n_components=self.latent_factors, random_state=None,
                                     power_iteration_normalizer='QR', n_iter=100)

        print('computing SVD expected urm ...')
        # self.u = sp.csr_matrix(u)
        # self.v = sp.csr_matrix(v)
        s = sp.diags(s)
        # self.s = sp.csr_matrix(s)
        self.X = u
        self.Y = s.dot(v)

        if verbose:
            print('SVD Matrix Factorization training computed in {:.2f} minutes'
                  .format((time.time() - start_time) / 60))

    def set_configuration(self):

        """ Sets the configuration of the current algorithm """

        now = datetime.datetime.now()
        self.configuration_txt = "SVD MF configuration {}:\n" \
                                 "date    ==> {}\n" \
                                 "model   ==>  latent_factors={}; scipy={}\n".format(SVDRecommender.N_CONFIG, now,
                                                                                     self.latent_factors, self.scipy)
        # increment the number of configuration
        SVDRecommender.N_CONFIG += 1
