import datetime
import time
from models.recommender import Recommender
from support.cosinesimilarity import CosineSimilarity


#################################################################
############# ITEM-BASED COLLABORATIVE FILTERING ################
#################################################################


class ItemBasedCFRecommender(Recommender):

    """ ITEM-BASED COLLABORATIVE FILTERING """

    N_CONFIG = 0

    def __init__(self, train, test, validation, targets, subfolder="../", log_filename='itembasedcf_config.txt'):
        super(ItemBasedCFRecommender, self).__init__(train, test, validation, targets, subfolder, log_filename)
        self.configuration_txt = "ITEM-BASED COLLABORATIVE FILTERING"

    def fit(self, similarity='asymmetric', topk=200, shrink=7, asymmetric_alpha=0.43, verbose=True):

        """ Fits Item-based collaborative filtering model"""

        self.similarity = similarity
        self.topk = topk
        self.shrink = shrink
        self.asymmetric_alpha = asymmetric_alpha

        start_time = time.time()

        sim_obj = CosineSimilarity(dataMatrix=self.URM_train, similarity=similarity, topK=topk,
                                   shrink=shrink, asymmetric_alpha=asymmetric_alpha)
        self.W = sim_obj.compute_similarity()

        self.predicted_URM = self.URM_train.dot(self.W)

        self.set_configuration()

        if verbose:
            print('ITEM-BASED COLLABORATIVE FILTERING model computed in {:.2f} minutes'.format((time.time() - start_time) / 60))

    def set_configuration(self):

        """ Sets the configuration of the current algorithm """

        now = datetime.datetime.now()
        self.configuration_txt = "ITEM-BASED CF configuration {}:\n" \
                                 "date    ==> {}\n" \
                                 "model   ==>  similarity={}; topk={}; shrink={}; " \
                                 "asymmetric_alpha={}\n".format(ItemBasedCFRecommender.N_CONFIG, now,
                                                                self.similarity, self.topk, self.shrink,
                                                                self.asymmetric_alpha)
        # increment the number of configuration
        ItemBasedCFRecommender.N_CONFIG += 1


