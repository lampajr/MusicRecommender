from models.slim.sslimrmse import SSLIMRMSERecommender
from support.dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset(split_traintestvalidation=(0.8, 0.2, 0.0))
    dataset.load_data()


    recommender = SSLIMRMSERecommender(train=dataset.URM_train, test=dataset.URM_test,
                                       validation=None, targets=dataset.target_playlists,
                                       albums=dataset.ICM_albums, artists=dataset.ICM_artists)
    recommender.fit()
    recommender.add_side_information()
    recommender.train()
    #recommender.evaluate()
    recommender.generate_recommendation(pathname='../outputs/slimrmse1.csv')