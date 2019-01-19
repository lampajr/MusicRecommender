from models.slim.slimbpr import *
from support.dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset(split_traintestvalidation=(0.8, 0.2, 0.0))
    dataset.load_data()

    recommender = SLIMBPRRecommender(dataset.URM_train, test=dataset.URM_test,
                                           validation=None, targets=dataset.target_playlists)
    recommender.fit()
    recommender.train()
    recommender.evaluate()
