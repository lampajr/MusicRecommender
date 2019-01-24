import datetime
import sys
import time
from models.recommender import Recommender
from support.cosinesimilarity import update_similarity_matrix_topk, CosineSimilarity

import numpy as np
import scipy.sparse as sp

import theano
import theano.sparse
import theano.tensor as T
from support.sampling import BPRSampling
from support.utility import check_matrix


################################################################
############## SLIM BPR SGD RECOMMENDER SYSTEM #####################
################################################################


def sigmoid(x):

    """ Logistic sigmoid function """

    return 1 / (1 + np.exp(-x))


class SLIMBPRRecommender(Recommender):

    """ SLIM BPR SGD RECOMMENDER SYSTEM """

    N_CONFIG = 0

    def __init__(self, train, test, validation, targets, log_filename='slimbpr_config.txt'):
        super(SLIMBPRRecommender, self).__init__(train, test, validation, targets, log_filename)
        self.configuration_txt = "SLIM BPR RECOMMENDER SYSTEM"

        self.BPR_sampling = BPRSampling(data=self.URM_train)

    # Override
    def compute_predicted_ratings(self, playlist_id):

        """ Compute predicted ratings for a given playlist in case of
            matrix factorization algorithm """

        return self.URM_train[playlist_id].dot(self.W).toarray().ravel()

    def fit(self, epochs=1, learning_rate=0.05, pos_lambda=0.005, neg_lambda=0.005, topk=200, normalize=False):

        """ Fits a SLIM BPR model using SGD """

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.topk = topk
        self.normalize = normalize
        #self.num_samples = self.URM_train.nnz
        self.num_samples = int(100 * self.n_playlists ** 0.5)


    def train(self, verbose=True):

        """ Train the SGD model"""

        start_time = time.time()

        # initialize the weight matrix
        # self.W = sp.csr_matrix((self.n_tracks, self.n_tracks), dtype=np.float32)
        sim_obj = CosineSimilarity(dataMatrix=self.URM_train, similarity='cosine', topK=500)
        self.W = sim_obj.compute_similarity()

        self.W = self.W.tolil()

        for ep in range(self.epochs):
            epoch_start_time = time.time()

            self.__epoch_iteration(ep=ep, verbose=False)

            if verbose:
                print('Epoch {} of {} --> computed in {:.2f} minutes'.format(ep+1, self.epochs,
                                                                         (time.time() - epoch_start_time) / 60))
                sys.stdout.flush()



        if self.topk is not None:
            self.W = update_similarity_matrix_topk(data=self.W, k=self.topk)
        else:
            self.W = check_matrix(data=self.W, format='csr', dtype=np.float32)



        if verbose:
            print('SLIM BPR TRAINING with SGD computed in {:.2f} minutes'.format((time.time() - start_time) / 60))


    def __epoch_iteration(self, ep, verbose=True):

        """ Single epoch of training """

        for s in range(self.num_samples):

            p_id, pos_track, neg_track = self.BPR_sampling.generate_sample()

            seen_tracks = self.get_relevant_items(p_id=p_id, set='train')

            # updates the W columns
            W_i = self.W[pos_track, seen_tracks].toarray().ravel()
            W_j = self.W[neg_track, seen_tracks].toarray().ravel()

            epij = W_i - W_j
            epij = np.sum(epij)

            sigm = sigmoid(epij)
            gradient = 1. - sigm

            delta_i = gradient - (self.pos_lambda * W_i)
            delta_j = - gradient - (self.neg_lambda * W_j)

            # updates positive track similarities
            updates = W_i + (self.learning_rate * delta_i)
            self.W[pos_track, seen_tracks] = updates
            self.W[pos_track, pos_track] = 0.0

            # updates negative track similarities
            updates = W_j + (self.learning_rate * delta_j)
            self.W[neg_track, seen_tracks] = updates
            self.W[neg_track, neg_track] = 0.0

            if verbose and s % 100000 == 0:
                print('Epoch {}: processed {} samples ({:.2f}% complete)'.format(ep, s,
                                                                                 (s/self.num_samples * 100)))
                sys.stdout.flush()

    def set_configuration(self):

        """ Sets the configuration of the current algorithm """

        now = datetime.datetime.now()
        self.configuration_txt = "SLIM BPR with SGD configuration {}:\n" \
                                 "date    ==> {}\n" \
                                 "model   ==>  epochs={}; learning_rate={}; pos_lambda={}; " \
                                 "neg_lambda={}; topk={}; num_samples={}\n".format(SLIMBPRRecommender.N_CONFIG, now,
                                                                                   self.epochs, self.learning_rate,
                                                                                   self.pos_lambda, self.neg_lambda,
                                                                                   self.topk, self.num_samples)

        # increment the number of configuration
        SLIMBPRRecommender.N_CONFIG += 1



#################################################################
############# SLIM BPR THEANO RECOMMENDER SYSTEM ################
#################################################################


class SLIMBPRTheanoRecommender(Recommender):

    """ SLIM BPR THEANO RECOMMENDER SYSTEM """

    N_CONFIG = 0

    def __init__(self, train, test, validation, targets, log_filename='slimbprtheano_config.txt'):
        super(SLIMBPRTheanoRecommender, self).__init__(train, test, validation, targets, log_filename)
        self.configuration_txt = "SLIM BPR THEANO RECOMMENDER SYSTEM"

        self.BPR_sampling = BPRSampling(data=self.URM_train)


    # Override
    def compute_predicted_ratings(self, playlist_id):

        """ Compute predicted ratings for a given playlist in case of
        matrix factorization algorithm """

        #return np.dot(self.URM_train[playlist_id], self.W)
        return self.predicted_URM[playlist_id]

    def fit(self, epochs=1, learning_rate=0.05, pos_lambda=0.0025, neg_lambda=0.00025, topk=None, normalize=False):

        """ Fits a SLIM BPR model using Theano """

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.topk = topk
        self.normalize = normalize
        self.num_samples = self.URM_train.nnz
        #self.num_samples = 50000

        # configure theano model
        self.__configure_theano()

        # generate the train model function
        self.__generate_train_model_function()

    def train(self, verbose=True):

        """ Train the theano model"""

        start_time = time.time()

        for ep in range(self.epochs):
            epoch_start_time = time.time()

            self.__epoch_iteration(ep=ep, verbose=verbose)

            if verbose:
                print('Epoch {} of {} --> computed in {:.2f} minutes'.format(ep+1, ep,
                                                                         (time.time() - epoch_start_time) / 60))
                sys.stdout.flush()

        self.W = self.S.get_value().T

        if self.topk is not None:
            self.W = update_similarity_matrix_topk(data=sp.csr_matrix(self.W), k=self.topk)

        self.predicted_URM = self.URM_train.dot(self.W)

        self.__clear_theano_train_data()

        if verbose:
            print('THEANO SLIM BPR TRAINING computed in {:.2f} minutes'.format((time.time() - start_time) / 60))


    def __epoch_iteration(self, ep, verbose=True):

        """ Single epoch of training """

        for s in range(self.num_samples):

            p_id, pos_track, neg_track = self.BPR_sampling.generate_sample()

            self.train_model(p_id, pos_track, neg_track)

            if verbose and s % 5000 == 0:
                print('Epoch {}: processed {} samples ({:.2f}% complete)'.format(ep, s,
                                                                                 (s/self.num_samples * 100)))
                sys.stdout.flush()

    def __configure_theano(self):

        """ Configures the Theano model """

        # compile highly optimized code
        theano.config.mode = 'FAST_RUN'

        # set default float32
        theano.config.floatX = 'float32'

        # enable multicore
        theano.config.openmp = 'true'

        theano.config.on_used_input = 'ignore'

    def __generate_train_model_function(self):

        """ Define the update rules to be used in the training phase """

        p = theano.tensor.scalar('p', dtype='int32')
        i = theano.tensor.scalar('i', dtype='int32')
        j = theano.tensor.scalar('j', dtype='int32')

        localS = np.random.random((self.n_tracks, self.n_tracks)).astype('float32')
        localS[np.arange(0, self.n_tracks), np.arange(0, self.n_tracks)] = 0.0

        self.S = theano.shared(localS, name='S')

        # set the URM boolean mask
        self.URM_mask = theano.shared((self.URM_train.toarray()>0).astype('int8'), name='URM')

        x_pi = self.S[i, :]
        x_pj = self.S[j, :]

        # the difference is computed over the whole row, not only over the playlist_interacted tracks
        x_pij = x_pi - x_pj

        sigmoid = T.nnet.sigmoid(-x_pij)

        # select only the tracks that playlist already has interacted with
        tracks_to_update = self.URM_mask[p:p+1, 0:self.n_tracks]

        #tracks_to_update = theano.sparse.dense_from_sparse(tracks_to_update)
        tracks_to_update = T.reshape(tracks_to_update, [self.n_tracks])

        # don't update track i, set all playlist-pos_track to false
        tracks_to_update = T.set_subtensor(tracks_to_update[i], 0)

        delta_i = sigmoid - self.pos_lambda * self.S[i]
        delta_j = - sigmoid - self.neg_lambda * self.S[j]

        # since a shared variable may be the target of only one update rule
        # all the required updates are chained inside a subtensor
        update_chain = self.S
        update_chain = T.inc_subtensor(update_chain[i], (self.learning_rate * delta_i) * tracks_to_update)

        tracks_to_update = T.set_subtensor(tracks_to_update[i], 1)
        tracks_to_update = T.set_subtensor(tracks_to_update[j], 0)

        update_chain = T.inc_subtensor(update_chain[j], (self.learning_rate * delta_j) * tracks_to_update)

        updates = [(self.S, update_chain)]

        # create and compile the train function
        self.train_model = theano.function(inputs=[p, i, j], updates=updates)

    def __clear_theano_train_data(self):

        """ Clear unused data """

        del self.S
        del self.URM_mask


