from models.matrixfactorization.svd import *
from support.dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset(split_traintestvalidation=(0.8, 0.2, 0.0))
    dataset.load_data()

    recommender = SVDRecommender(train=dataset.URM_train, test=dataset.URM_test, validation=None,
                                 targets=dataset.target_playlists)
    recommender.fit(latent_factors=1, scipy=False)
    recommender.train()
    recommender.evaluate()
