from models.hybrid.hybrid import HybridRecommender
from support.dataset import Dataset

if __name__ == '__main__':

    evaluation = True
    output_file = '../outputs/hybrid5_0_0_0e8_1_0e345_0e015.csv'

    if evaluation:
        dataset = Dataset(split_traintestvalidation=(0.8, 0.2, 0.0), subfolder="input/")
    else:
        dataset = Dataset(split_traintestvalidation=(1.0, 0.0, 0.0))

    dataset.load_data()

    recommender = HybridRecommender(train=dataset.URM_train, test=dataset.URM_test,
                                    validation=None, targets=dataset.target_playlists,
                                    albums=dataset.ICM_albums, artists=dataset.ICM_artists,
                                    duration=dataset.ICM_duration, subfolder=None)
    recommender.fit()
    recommender.set_weights()

    if evaluation:
        recommender.evaluate()
    else:
        recommender.generate_recommendation(pathname=output_file)
