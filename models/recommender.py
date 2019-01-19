import time

from support.metrics import *
from support.utility import read_data, create_csv


##########################################################
################ ABSTRACT RECOMMENDER ####################
##########################################################

class Recommender(object):

    """ ABSTRACT RECOMMENDER """

    def __init__(self, train, test, validation, targets, log_filename=None):
        super(Recommender, self).__init__()
        # URMs
        self.URM_train = train
        self.URM_test = test
        self.URM_validation = validation

        # playlists for which we've to provide recommendation
        self.target_playlists = targets

        # ICMs
        self.ICM_albums = None
        self.ICM_artists = None
        self.ICM_duration = None

        # predicted URM
        self.predicted_URM = None

        # weights matrix
        self.W = None

        # log filename
        self.log_filename = log_filename
        self.log_subfolder = "../logs/"

        # configuration description
        self.configuration_txt = "ABSTRACT RECOMMENDER"

        self.n_playlists, self.n_tracks = self.URM_train.shape

        if self.n_playlists != 50446 or self.n_tracks != 20635:
            exit('Inconsistent shape found!!!')

    def __str__(self):

        """ String representation of the class """

        return self.configuration_txt

    def recommend(self, playlist_id, at=10, remove_seen=True):

        """
        Provides a list of 'at' recommended items for the given playlist
        :param playlist_id: id for which provide recommendation
        :param at: how many items have to be recommended
        :param remove_seen: if remove already seen items
        :return: recommended items list
        """

        predicted_ratings = self.compute_predicted_ratings(playlist_id=playlist_id)
        ordered_tracks = np.flip(np.argsort(predicted_ratings))

        if remove_seen:
            unseen_items = self.__remove_seen(p_id=playlist_id, tracks=ordered_tracks)
            recommended_items = unseen_items[:at]
        else:
            recommended_items = ordered_tracks[:at]

        return recommended_items

    def evaluate(self, set='test', at=10, all=True, verbose=True):

        """
        evaluates the algorithm
        :param set: from which set compute evaluation
        :param at: number of items to recommend
        :param all: if True iterates over all playlists, only on the targets otherwise
        :return: return a tuple of metric evaluation over all playlist
        """

        if all:
            num_playlists = self.n_playlists
            playlists = np.arange(start=0, stop=num_playlists)
        else:
            num_playlists = len(self.target_playlists)
            playlists = self.target_playlists

        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_map = 0.0

        count = 0
        start_time = time.time()

        if verbose:
            print('Evaluation process started ...')

        for play_id in playlists:
            recommended_items = self.recommend(playlist_id=play_id, at=at)

            cur_pre, cur_rec, cur_map = self.__evaluate_playlist(playlist_id=play_id,
                                                                 recommended_items=recommended_items,
                                                                 set=set)

            cumulative_precision += cur_pre
            cumulative_recall += cur_rec
            cumulative_map += cur_map

            if verbose and count % 2500 == 0:
                print('Evaluating... {:.2f}% complete'.format(count / num_playlists * 100))

            count += 1

        precision_value = cumulative_precision / num_playlists
        recall_value = cumulative_recall / num_playlists
        map_value = cumulative_map / num_playlists

        # update the configuration of the system and log it
        self.update_configuration(precision_value, recall_value, map_value)
        self.log_configuration()

        if verbose:
            print('Evaluation process computed in {:.2f} minutes'.format((time.time() - start_time) / 60))
            print('----------------------------------------')
            print('Precision => {:.6f}'.format(precision_value))
            print('Recall => {:.6f}'.format(recall_value))
            print('MAP@{} => {:.6f}'.format(at, map_value))

    def set_configuration(self):
        pass

    def update_configuration(self, precision_value, recall_value, map_value):

        """ Updates the configuration of the current algorithm with metrics results """

        self.configuration_txt += "metrics ==>  precision={:.6f}; " \
                                  "recall={:.6f}; map={:.6f}\n".format(precision_value, recall_value, map_value)

        self.configuration_txt += "------------------------------------------------\n"

    def generate_recommendation(self, pathname, at=10, verbose=True):

        """ Provides recommendation for all the target playlists and generates
            the csv file containing them """

        result_df = read_data(filename='sample_submission.csv')
        playlists = result_df['playlist_id'].copy()

        count = 0
        start_time = time.time()

        if verbose:
            print('Recommendation process started...')


        for play_id in playlists:
            recommended_items = self.recommend(playlist_id=play_id, at=at)
            result_df.set_value(index=count, col='track_ids', value=' '.join(str(x) for x in recommended_items))
            if verbose and count % 500 == 0:
                print('Recommending... {:.2f}%'.format(count / len(playlists) * 100))
            count += 1

        create_csv(data=result_df, pathname=pathname)

        if verbose:
            print('Recommendation process computed in {:.2f} minutes'.format((time.time() - start_time) / 60))
            print('CSV file created at {}'.format(pathname))


    def log_configuration(self, verbose=True):

        """ Logs information about the current recommender into a txt file """

        pathname = self.log_subfolder + self.log_filename
        if verbose:
            print('Saving current configuration at {}'.format(pathname))
        with open(pathname, 'a') as f:
            f.write(self.configuration_txt)

    ######## PRIVATE METHODS #########

    def __evaluate_playlist(self, playlist_id, recommended_items, set='test'):

        """
        Evaluates the recommender algorithm on a single playlist
        :param playlist_id:
        :param recommended_items: list of recommendation provided by the algorithm
        :return: a tuple of metrics evaluation (precision, recall, map)
        """

        relevant_items = self.get_relevant_items(p_id=playlist_id, set=set)

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        return precision(is_relevant), recall(is_relevant, relevant_items), map_at(is_relevant, relevant_items)

    def __get_top_popular_items(self):

        """ Provides a list of track ordered by popularity """

        item_popularity = (self.URM_train > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()

        # Sort the array but return an array of its indexes
        top_items = np.argsort(item_popularity)
        return np.flip(top_items, axis=0)

    def __remove_seen(self, p_id, tracks):

        """
        Removes already seen items from the recommended list
        :param p_id: playlist id
        :param tracks: recommended list
        :return: filtered recommended list
        """

        relevant_items = self.get_relevant_items(p_id=p_id, set='train')
        unseen_items_mask = np.in1d(tracks, relevant_items, assume_unique=True, invert=True)
        return tracks[unseen_items_mask]

    def get_relevant_items(self, p_id, set='test'):

        """
        Provides the relevant items for a given playlist drawing from a specific set
        :param p_id: playlist id
        :param set: tells from which set retrieves the relevant items
        :return: relevant items
        """

        if set == 'train':
            return self.URM_train[p_id].indices
        elif set == 'test':
            return self.URM_test[p_id].indices
        else:
            return self.URM_validation[p_id].indices

    def compute_predicted_ratings(self, playlist_id):

        """ Compute the predicted ratings for a given playlist """

        return self.predicted_URM[playlist_id].toarray().ravel()

    ####### GETTER METHODS #########

    def get_W(self):
        return self.W

    def get_predicted_URM(self):
        return self.predicted_URM


##########################################################
####### ABSTRACT MATRIX FACTORIZATION RECOMMENDER ########
##########################################################

class MatrixFactorizationRecommender(Recommender):

    """ ABSTRACT MATRIX FACTORIZATION RECOMMENDER """

    def __init__(self, train, test, validation ,targets, log_filename=None):
        super(MatrixFactorizationRecommender, self).__init__(train, test, validation, targets, log_filename)
        self.X = None  # playlist x latent_factors
        self.Y = None  # tracks x latent_factors

    def compute_predicted_ratings(self, playlist_id):

        """ Compute predicted ratings for a given playlist in case of
        matrix factorization algorithm """

        return np.dot(self.X[playlist_id], self.Y.T)

    ##### GETTER METHODS ######

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y
