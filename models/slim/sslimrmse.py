import datetime
import sys
import time
from models.recommender import Recommender
from support.utility import check_matrix
import scipy.sparse as sp
import numpy as np
from sklearn.linear_model import ElasticNet

##########################################################
############# SSLIM RMSE RECOMMENDER SYSTEM ##############
##########################################################

class SSLIMRMSERecommender(Recommender):

    """ SLIM RMSE RECOMMENDATION SYSTEM ALGORITHM
        slim implementation that minimizes the Root Mean Squared Error (RMSE)
        using the ElasticNet class provided by the Sklearn module.
        there is an option to add side information to the algorithm """

    N_CONFIG = 0

    def __init__(self, train ,test, validation, targets, albums=None, artists=None, duration=None,
                 subfolder="../", log_filename="sslimrmse_config.txt"):
        super(SSLIMRMSERecommender, self).__init__(train, test, validation, targets, subfolder, log_filename)
        self.ICM_albums = albums
        self.ICM_artists = artists
        self.ICM_duration = duration
        self.configuration_txt = "SSLIM RMSE RECOMMENDER SYSTEM"

    # alpha = 0.0002
    def fit(self, alpha=0.0005, l1_ratio=0.029126214, topk=900, positive_only=True):

        """ Fits the ElasticNet model """

        self.alpha = alpha
        #self.l1_penalty = l1_penalty
        #self.l2_penalty = l2_penalty
        #self.l1_ratio = l1_penalty / (l1_penalty + l2_penalty)
        self.l1_ratio = l1_ratio
        self.topk = topk
        self.positive_only = positive_only

        self.model = ElasticNet(alpha=alpha,
                                l1_ratio=self.l1_ratio,
                                positive=positive_only,
                                fit_intercept=False,
                                warm_start=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        # the matrix that has to be learnt
        self.A = check_matrix(data=self.URM_train, format='csc', dtype=np.float32)

    def add_side_information(self, beta=1.0, gamma=1.0, delta=None):

        """ Adds side information to the algorithm, implementing the so called SSLIM """

        self.beta, self.gamma, self.delta = beta, gamma, delta

        if beta is not None:
            self._stack(self.ICM_albums.T, beta)

        if gamma is not None:
            self._stack(self.ICM_artists.T, gamma)

        if delta is not None:
            self._stack(self.ICM_duration.T, delta)

    def train(self, verbose=True):

        """ Trains the ElasticNet model """

        A = self.A

        # we'll construct the W matrix incrementally
        values, rows, columns = [], [], []

        training_start_time = time.time()
        batch_start_time = training_start_time

        # iterates over all tracks in the URM and compute the W column for each of them
        # self.n_tracks
        for track in range(self.n_tracks):

            # consider the current column track as the target for the training problem
            y = A[:, track].toarray()

            # set to zero the current column in A
            startptr = A.indptr[track]
            endptr = A.indptr[track+1]

            # save the data of the current column in a temporary variable
            data_t = A.data[startptr:endptr].copy()

            A.data[startptr:endptr] = 0.0

            # fit the ElasticNet model
            self.model.fit(A, y)

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            relevant_items_partition = (-self.model.coef_).argpartition(self.topk)[0:self.topk]
            # - Sort only the relevant items
            relevant_items_partition_sorting = np.argsort(-self.model.coef_[relevant_items_partition])
            # - Get the original item index
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            # keep only non-zero values
            not_zeros_mask = self.model.coef_[ranking] > 0.0
            ranking = ranking[not_zeros_mask]

            values.extend(self.model.coef_[ranking])
            rows.extend(ranking)
            columns.extend([track] * len(ranking))

            # finally, replace the original values of the current track column
            A.data[startptr:endptr] = data_t

            if track % 1000 == 0 and verbose:
                print(
                    "Processed {} overall ( {:.2f}% ), previous batch in {:.2f} seconds. Columns per second: {:.0f}".format(
                        track,
                        100.0 * float(track) / self.n_tracks,
                        (time.time() - batch_start_time),
                        float(track) / (time.time() - training_start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                batch_start_time = time.time()

        # generate the sparse weight matrix
        self.W = sp.csr_matrix((values, (rows, columns)), shape=(self.n_tracks, self.n_tracks), dtype=np.float32)

        self.predicted_URM = self.URM_train.dot(self.W)

        self.set_configuration()

        if verbose:
            print('SLIM RMSE training computed in {:.2f} minutes'.format((time.time() - training_start_time) / 60))

    def set_configuration(self):

        """ Sets the configuration of the current algorithm """

        now = datetime.datetime.now()
        self.configuration_txt = "SSLIM RMSE configuration {}:\n" \
                                 "date    ==>  {}\n" \
                                 "model   ==>  alpha={}; l1_ratio={:.6f}; " \
                                 "topk={}; positive_only={}\n" \
                                 "side    ==>  albums_param={}; artists_param={}; " \
                                 "duration_param={}\n".format(SSLIMRMSERecommender.N_CONFIG, now,
                                                              self.alpha, self.l1_ratio,
                                                              self.topk, self.positive_only,
                                                              self.beta, self.gamma, self.delta)
        # increment the number of configuration
        SSLIMRMSERecommender.N_CONFIG += 1


    ######### PRIVATE METHODS ###############

    def _stack(self, to_stack, param, format='csc'):

        """
        Stacks a new sparse matrix under the A matrix used for training
        :param to_stack: sparse matrix to add
        :param param: regularization
        :param format: default 'csc'
        """

        tmp = check_matrix(to_stack, 'csc', dtype=np.float32)
        tmp = tmp.multiply(param)
        self.A = sp.vstack((self.A, tmp), format=format, dtype=np.float32)
