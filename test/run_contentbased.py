from models.contentbasedfiltering.contentbasedfiltering import CBFRecommender
from support.dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset(split_traintestvalidation=(0.8, 0.2, 0.0))
    dataset.load_data()

    recommender = CBFRecommender(dataset.URM_train, test=dataset.URM_test,
                                 validation=None, targets=dataset.target_playlists,
                                 albums=dataset.ICM_albums, artists=dataset.ICM_artists,
                                 duration=dataset.ICM_duration)
    recommender.fit()
    recommender.evaluate()