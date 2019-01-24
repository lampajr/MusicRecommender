from models.hybrid.hybrid import HybridRecommender
from support.dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset(split_traintestvalidation=(0.8, 0.2, 0.0), subfolder="input/")
    #dataset = Dataset(split_traintestvalidation=(1.0, 0.0, 0.0))
    dataset.load_data()

    recommender = HybridRecommender(train=dataset.URM_train, test=dataset.URM_test,
                                    validation=None, targets=dataset.target_playlists,
                                    albums=dataset.ICM_albums, artists=dataset.ICM_artists,
                                    duration=dataset.ICM_duration, subfolder=None)
    recommender.fit(k=None)
    recommender.set_weights()
    #recommender.generate_recommendation(pathname='../outputs/hybrid5_0_0_0e8_1_0e345_0e015.csv')
    recommender.evaluate()
