import datetime
import time

from models.recommender import MatrixFactorizationRecommender
import numpy as np


def non_zeros(m, row):

    """ Returns an tuple iterator (tracks, data) """

    for index in range(m.indptr[row], m.indptr[row + 1]):
        yield m.indices[index], m.data[index]

def least_squares_cg(Cui, X, Y, lambda_val, cg_steps=3):

    """ Computes the least square solution using the conjugate gradient """

    users, features = X.shape

    YtY = Y.T.dot(Y) + lambda_val * np.eye(features)

    for u in range(users):

        x = X[u]
        r = -YtY.dot(x)

        for i, confidence in non_zeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]

        p = r.copy()
        rsold = r.dot(r)

        for it in range(cg_steps):
            Ap = YtY.dot(p)
            for i, confidence in non_zeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap

            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x


class ALSMFRecommender(MatrixFactorizationRecommender):

    """ ALTERNATING LEAST SQUARE MATRIX FACTORIZATION RECOMMENDER SYSTEM ALGORITHM """

    N_CONFIG = 0

    def __init__(self, train, test, validation, targets, subfolder="../", log_filename='alsmf_config.txt'):
        super(ALSMFRecommender, self).__init__(train, test, validation, targets, subfolder, log_filename)

    def fit(self, alpha=40, lambda_val=1.8, latent_factors=460, iterations=20):

        """ Fits the ALS MF model """

        self.alpha = alpha
        self.lambda_val = lambda_val
        self.latent_factors = latent_factors
        self.iterations = iterations
        self.Cui = self.URM_train.multiply(alpha).astype('double')

        self.set_configuration()

    def train(self, verbose=True):

        """ Trains the ALS MF model """

        start_time = time.time()

        if verbose:
            print('ALS training started...')

        user_size, item_size = self.Cui.shape
        self.X = np.random.rand(user_size, self.latent_factors)
        self.Y = np.random.rand(item_size, self.latent_factors)

        Cui, Ciu = self.Cui.tocsr(), self.Cui.T.tocsr()

        for iteration in range(self.iterations):
            iter_start_time = time.time()

            least_squares_cg(Cui, self.X, self.Y, self.lambda_val)
            least_squares_cg(Ciu, self.Y, self.X, self.lambda_val)
            print('iteration {} of {} --> computed in {:.2f} minutes'.format(iteration + 1,
                                                                             self.iterations,
                                                                             (time.time() - iter_start_time) / 60))

        if verbose:
            print('ALS Matrix Factorization training computed in {:.2f} minutes'
                  .format((time.time() - start_time) / 60))

    def set_configuration(self):

        """ Sets the configuration of the current algorithm """

        now = datetime.datetime.now()
        self.configuration_txt = "ALS MF configuration {}:\n" \
                                 "date    ==> {}\n" \
                                 "model   ==>  alpha={}; lambda={}; " \
                                 "latent_factors={}; iterations={}\n".format(ALSMFRecommender.N_CONFIG, now,
                                                                             self.alpha, self.lambda_val,
                                                                             self.latent_factors, self.iterations)
        # increment the number of configuration
        ALSMFRecommender.N_CONFIG += 1


############################################
############ IMPLICIT ALS MF ###############
############################################

import implicit as impl

class ImplicitALSRecommender(ALSMFRecommender):

    """ ALS implementation using the implicit library """

    def __init__(self, train ,test, validation, targets, subfolder="../", log_filename='implicitalsmf_config.txt'):
        super(ImplicitALSRecommender, self).__init__(train, test, validation, targets, subfolder, log_filename)
        self.model = None

    def train(self, model_name='als', verbose=True):

        """ train the ALS model using the implicit module """

        start_time = time.time()

        # creates ALS model
        if model_name == 'als':
            self.model = impl.als.AlternatingLeastSquares(factors=self.latent_factors, regularization=self.lambda_val,
                                                          iterations=self.iterations)
        elif model_name == 'nmslibals':
            self.model = impl.approximate_als.NMSLibAlternatingLeastSquares(factors=self.latent_factors,
                                                                            regularization=self.lambda_val,
                                                                            iterations=self.iterations)
        elif model_name == 'faissals':
            self.model = impl.approximate_als.FaissAlternatingLeastSquares(factors=self.latent_factors,
                                                                           regularization=self.lambda_val,
                                                                           iterations=self.iterations)
        elif model_name == 'annoyals':
            self.model = impl.approximate_als.AnnoyAlternatingLeastSquares(factors=self.latent_factors,
                                                                           regularization=self.lambda_val,
                                                                           iterations=self.iterations)
        else:
            exit('Invalid model name')

        # fit the ALS model
        # since the model is expecting a item-user matrix we need to pass the transpose of URM
        A = self.URM_train.T.copy()
        self.model.fit(A)

        # gets the results of the training
        self.X = self.model.user_factors
        self.Y = self.model.item_factors

        if verbose:
            print("IMPLICIT ALS training computed in {:.2f} seconds".format(time.time() - start_time))
