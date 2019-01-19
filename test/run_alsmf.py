from models.matrixfactorization.alsmf import *
from support.dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset(split_traintestvalidation=(0.8, 0.2, 0.0))
    dataset.load_data()

    recommender = ImplicitALSRecommender(train=dataset.URM_train, test=dataset.URM_test,
                                         validation=None, targets=dataset.target_playlists)
    recommender.fit(latent_factors=350, iterations=20, lambda_val=10, alpha=40)
    recommender.train(model_name='annoyals')
    recommender.evaluate()
