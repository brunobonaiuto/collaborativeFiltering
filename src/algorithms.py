''' 
Importing the required libraries
* pandas: for data manipulation and analysis
* numpy: for scientific computing
* matplotlib: for plotting graphs
* seaborn: for plotting graphs

* surprise: for building and analyzing recommender systems, this librery contains the based line of the algorithms
    * Reader: to parse the file containing the ratings
    * Dataset: to load the dataset
    * KNNBasic: to create a basic collaborative filtering algorithm, it's the basic version of the KNN algorithm
    * accuracy: to calculate the RMSE and MAE of the algorithm
    * PredictionImpossible: to raise an exception when a prediction is impossible
    * train_test_split: to split the dataset into train and test sets

'''

#Libraries
import pandas as pd
from surprise import Reader, Dataset, KNNBasic, KNNWithMeans, KNNWithZScore, SVD, SVDpp, NMF, SlopeOne, CoClustering, accuracy, PredictionImpossible
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold
from random import shuffle
from collections import defaultdict

class DynamicAlgo:
    '''
    * The DynamicAlgo class is wrapper around the other class of surprise, which is called dynamically at the instantiation of the class.
    * The algorithm_class parameter is the class of the real-algorithm that we want to use, it can be KNNBasic, KNNWithMeans, KNNWithZScore, SVD, SVDpp, NMF, SlopeOne, CoClustering
    * The DynamicAlgo class is desing to create dynamically a new class using the built-in type() function.
        * The type() function takes three parameters:
            * DynamicAlgorithm: the name of the algorithm to inherit from
            * (algorithm_class,): the class of the algorithm
            * {'k': k, 'sim_options': sim_options, 'bsl_options': bsl_options}: the parameters of the algorithm
    '''

    def __init__(self, algorithm_class, k = None, biased = None, sim_options={}, bsl_options={}):
        # for knns
        self.k = k
        # for svds
        self.biased = biased
        self.algorithm = type('DynamicAlgorithm', (algorithm_class,), {'k': k, 'sim_options': sim_options, 'bsl_options': bsl_options})()

    def create_reader(self, data):
        reader = Reader(rating_scale=(1, 5))
        self.data = Dataset.load_from_df(data[['userId', 'wine', 'rate']], reader)
        
    
    def fit (self):
        '''Divide the data into train and test manually '''
        #Frist shuffle the data
        raw_ratings = self.data.raw_ratings
        shuffle(raw_ratings)
        
        #75% of the data for training and the rest for testing
        threshold = int(.80 * len(raw_ratings))
        train_raw_ratings = raw_ratings[:threshold]
        test_raw_ratings = raw_ratings[threshold:]
        
        #Update the data object with train raw ratings
        self.data.raw_ratings = train_raw_ratings
        
        #Define a cross-validation iterator
        kf = KFold(n_splits=5, shuffle= True, random_state=42)
        
        train_rmse_list = []
        test_rmse_list = []
        
        for trainset_fold, testset_fold in kf.split(self.data):
            #Train and test algorithm.
            self.algorithm.fit(trainset_fold)
            train_prediction = self.algorithm.test(trainset_fold.build_testset())
            prediction = self.algorithm.test(testset_fold)

            #Error on training
            train_rmse = accuracy.rmse(train_prediction)
            train_rmse_list.append(train_rmse)

            #Error on testing
            test_rmse = accuracy.rmse(prediction)
            test_rmse_list.append(test_rmse)

        avg_train_rmse = sum(train_rmse_list) / len(train_rmse_list)
        avg_test_rmse = sum(test_rmse_list) / len(test_rmse_list)
        print("###############################################")
        print("The Results are: \n")
        print(f"Average RMSE on Training Set: {avg_train_rmse}")
        print(f"Average RMSE on Test Set: {avg_test_rmse}")

        #Predict ratings for all pairs (u, i) that are NOT in the training set.
        #Update the data object with test raw ratings
        self.data.raw_ratings = test_raw_ratings
        #adapting the Testset to be compatible with Surprise
        testset = self.data.construct_testset(self.data.raw_ratings)
        #Predicting the ratings for testset
        predictions = self.algorithm.test(testset)
        #RMSE
        test_rmse = accuracy.rmse(predictions)
        print(f"RMSE on Test Set on UNSEEN DATA is RMSE, : {test_rmse}")

        self.sim = self.algorithm.compute_similarities()
        self.bu, self.bi = self.algorithm.compute_baselines()
        return predictions
        
    def get_user_name(self, uid):
        """Return the name of a user from their id.
        Args:
            uid(int): The raw id of the user.
        Returns:
            The name of the user.
        """
        return self.algorithm.trainset.to_raw_uid(uid)
    
    def get_item_name(self, iid):
        """Return the name of an item from their id.
        Args:
            iid(int): The raw id of the item.
        Returns:
            The name of the item.
        """
        return self.algorithm.trainset.to_raw_iid(iid)
    
    def get_neighbors_uid(self, user_id, k=10):
        neighbor_ids = self.algorithm.get_neighbors(user_id, k=10)
        neighbor_names = [self.get_user_name(uid) for uid in neighbor_ids]
        return neighbor_names
    
    def get_neighbors_iid(self, item_id, k=10):
        neighbor_ids = self.algorithm.get_neighbors(item_id, k=10)
        neighbor_names = [self.get_item_name(iid) for iid in neighbor_ids]
        return neighbor_names
        
    def get_top_n_for_user(self, predictions,user_id, n=10):
        """Return the top-N recommendation for a user from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            n(int): The number of recommendation to output for each user. Default
                is 10.

        Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
            [(raw item id, rating estimation), ...] of size n.
        """

        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            if uid == user_id:
                top_n[uid].append((iid, est))

        # Then sort the predictions for the user and retrieve the k highest ones.
        user_ratings = top_n[user_id]
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[user_id] = user_ratings[:n]

        return top_n[user_id]
    
    def get_top_n_users_for_item(self, predictions, item_id, n=10):
        """Return the top-N users for a specific item from a set of predictions.

        Args:
            predictions(list of Prediction objects): The list of predictions, as
                returned by the test method of an algorithm.
            item_id: The id of the item for which to get the top-N users.
            n(int): The number of users to output for the item. Default is 10.

        Returns:
        A list of tuples:
            [(raw user id, rating estimation), ...] of size n.
        """

        # First map the predictions to each item.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            if iid == item_id:
                top_n[iid].append((uid, est))

        # Then sort the predictions for the item and retrieve the k highest ones.
        item_ratings = top_n[item_id]
        item_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[item_id] = item_ratings[:n]

        return top_n[item_id]

    def estimated(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        
        #Compute similarities between u and v, where v describes all other
        #users that have also rated item i.
        neighbors = [(v, self.sim[u, v]) for (v, r) in self.trainset.ir[i]]
        # Sort these neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        print("The 5 nearest neighbors of user", str(u), "are:")
        for v, sim_uv in neighbors[:5]:
            print(f"user {v} with sim {sim_uv:1.15f}")

        # ... Aaaaand return the baseline estimate anyway ;)
        bsl = self.trainset.global_mean + self.bu[u] + self.bi[i]
        return print(f"And the baseline estimate is: {bsl}")
    
    def get_Iu(self, uid):
        """Return the number of items rated by given user
        args:
          uid: the id of the user
        returns:
          the number of items rated by the user
        """
        try:
            return len(self.trainset.ur[self.trainset.to_inner_uid(uid)])
        except ValueError:  # user was not part of the trainset
            return 0

    def get_Ui(self, iid):
        """Return the number of users that have rated given item
        args:
          iid: the raw id of the item
        returns:
          the number of users that have rated the item.
        """
        try:
            return len(self.trainset.ir[self.trainset.to_inner_iid(iid)])
        except ValueError:
            return 0

    def inspect_predictions(self, predictions):
        print(f"uid means the user id and iid means the wine id\n")
        print(f"rui means the actual rating and est means the estimated rating\n")
        print(f"err means the error between the actual and the estimated rating\n")
        print(f"Iu means the number of items rated by given user\n")
        print(f"Ui means the number of users that have rated given item\n")
        # Create a dataframe with the predictions
        df_pred = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
        df_pred['Iu'] = df_pred.uid.apply(self.get_Iu)
        df_pred['Ui'] = df_pred.iid.apply(self.get_Ui)
        df_pred['err'] = abs(df_pred.est - df_pred.rui)
        return df_pred
    
    def get_accuracy(self, predictions, k=10, threshold=3.5):
        #Compute RMSE
        rmse = accuracy.rmse(predictions, verbose=True)
        #Compute MAE
        mae = accuracy.mae(predictions, verbose=True)
        # Compute MSE
        mse = accuracy.mse(predictions, verbose=True)

        #Compute precision and recall
        precisions, recalls = self.precision_recall_at_k(predictions, k=k, threshold=threshold)

        #Precision and recall can then be averaged over all users
        precision = sum(prec for prec in precisions.values()) / len(precisions)
        recall = sum(rec for rec in recalls.values()) / len(recalls)
        print(f'Precision: {precision:.2f}\nRecall: {recall:.2f}')

        #Count correct predictions
        correct = 0
        for uid, iid, true_r, est, _ in predictions:
            if round(est) == round(true_r):
                correct += 1

        #Compute accuracy
        accuracy_percentage = correct / len(predictions)
        accuracy_percentage = accuracy_percentage * 100
        print(f"the acc is {accuracy_percentage:.2f}")

        #Return a dictionary with the metrics
        return {'RMSE': rmse, 'MAE': mae, 'MSE': mse, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy_percentage}
    
      
    @staticmethod 
    def precision_recall_at_k(predictions, k=10, threshold=3.5):
        """Return precision and recall at k metrics for each user"""
        # First map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():
            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            # Precision@K: Proportion of recommended items that are relevant
            # When n_rec_k is 0, Precision is undefined. We here set it to 0.
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@K: Proportion of relevant items that are recommended
            # When n_rel is 0, Recall is undefined. We here set it to 0.
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
        return precisions, recalls
    
    def process_algorithm(algo, data, dataset, metrics_df):
        print(f"algorithm: {algo['algorithm_class']}")
        algorithms_instance = DynamicAlgo(**algo)
        algorithms_instance.create_reader(data)
        predictions = algorithms_instance.fit()
        algorithms_instance.get_accuracy(predictions)

        # metrics
        metrics = algorithms_instance.get_accuracy(predictions)
        metrics['Model'] = algo['algorithm_class'].__name__.split('.')[-1]
        if 'biased' in algo:
            metrics['Biased'] = algo['biased']
        if 'sim_options' in algo:
            metrics['Metric'] = algo['sim_options']['name']
            metrics['User_based'] = bool(algo['sim_options']['user_based'])
            metrics['k'] = algo['k']
        if 'bsl_options' in algo:
            metrics['Optimizer'] = algo['bsl_options']['method']
        metrics['DataSet'] = dataset
        # Append the dictionary to the existing DataFrame
        return pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])], ignore_index=True)

    
if __name__ == "__main__":
    import warnings
    # Ignore the specific FutureWarning
    warnings.filterwarnings('ignore')

    # Create an empty dataframe for KNN algorithms
    metrics_df = pd.DataFrame(
        columns=['Model','Metric','Optimizer','User_based', 'k',
                    'RMSE','MAE','MSE', 'Precision', 'Recall', 'Accuracy', 'DataSet'])
    
    # Create an empty dataframe for SVD, SVD++, NMF, SlopeOne, CoClustering algorithms
    metrics_df2 = pd.DataFrame(
        columns=['Model','Biased','RMSE','MAE','MSE', 'Precision', 'Recall', 'Accuracy', 'DataSet'])

    # List of datasets
    datasets = [
                # Oversampling data sets
                'upsampled_df_smote_auto', 
                'upsampled_df_smote_auto_distribution_kept',
                'df_oversamling',
                
                # Undersampling data sets
                'downsampled_df_random',
                ]
    
    # iterate over all datasets
    for dataset in datasets:
        data = pd.read_csv(f'/home/bbruno/all_here/python course/vinnie/data/cleaned_data/{dataset}.csv')

        ##### START OF KNN algorithms #####
        # Iterate over all the KNN algorithms
        for algorithm_class in [KNNBasic,KNNWithMeans, KNNWithZScore]:
            for sim in ['cosine', 'msd', 'pearson', 'pearson_baseline']:
                for user_based in [True, False]:
                    for bls in ['sgd', 'als']:
                        for k in [5, 10]:
                            algo = {'algorithm_class': algorithm_class, 'k': k, 'sim_options': {'name': sim,'user_based': user_based}, 
                                    'bsl_options':{'method': bls, 'learning_rate': 0.05, 'n_epochs':60, 'reg_u': 12 , 'reg_i': 5}}
                            metrics_df = DynamicAlgo.process_algorithm(algo, data, dataset, metrics_df)

        #save the dataframe
        metrics_df.to_csv('/home/bbruno/all_here/python course/vinnie/data/cleaned_data/metrics_df_knn.csv', index=False)
        print(metrics_df)
        ##### END OF KNN algorithms #####

        ##### START OF SVD, SVD++, NMF, SlopeOne, CoClustering algorithms #####
        # Iterate over all the algorithms
        # for algorithm_class in [SVD, NMF]:
        for algorithm_class in [SVD, NMF, SVDpp, SlopeOne, CoClustering]:
            # Iterate over the biased algorithms
            if algorithm_class == SVD or algorithm_class == NMF:
                for biased in [True, False]:
                    algo = {'algorithm_class': algorithm_class, 'biased': biased}
                    metrics_df2 = DynamicAlgo.process_algorithm(algo, data, dataset, metrics_df2)
            else: #iterate over the non-biased algorithms
                algo = {'algorithm_class': algorithm_class, 'biased': False}
                metrics_df2 = DynamicAlgo.process_algorithm(algo, data, dataset, metrics_df2)

        #save the dataframe
        metrics_df2.to_csv('/home/bbruno/all_here/python course/vinnie/data/cleaned_data/metrics_df_others.csv', index=False)
        print(metrics_df2)
        ##### END OF SVD, SVD++, NMF, SlopeOne, CoClustering algorithms #####
 
    






    # # iterate over all datasets
    # for dataset in datasets:
    #     data = pd.read_csv(f'/home/bbruno/all_here/python course/vinnie/data/cleaned_data/{dataset}.csv')
        
    #     # Iterate over all the algorithms
    #     for algorithm_class in [KNNBasic,KNNWithMeans, KNNWithZScore]:
    #         for sim in ['cosine', 'msd', 'pearson', 'pearson_baseline']:
    #             for user_based in [True, False]:
    #                 for bls in ['sgd', 'als']:
    #                     for k in [5, 10]:
    #                         algo = {'algorithm_class': algorithm_class, 'k': k, 'sim_options': {'name': sim,'user_based': user_based}, 
    #                                 'bsl_options':{'method': bls, 'learning_rate': 0.05, 'n_epochs':60, 'reg_u': 12 , 'reg_i': 5}}
                        
    #                         print(f"algorithm: {algo['algorithm_class']}")
    #                         algorithms_instance = DynamicAlgo(**algo)
    #                         algorithms_instance.create_reader(data)
    #                         predictions = algorithms_instance.fit()
    #                         algorithms_instance.get_accuracy(predictions)
                        
    #                         # metrics
    #                         metrics = algorithms_instance.get_accuracy(predictions)
    #                         metrics['Model'] = algo['algorithm_class'].__name__.split('.')[-1]
    #                         metrics['Metric'] = sim
    #                         metrics['Optimizer'] = bls
    #                         # Ensure 'User_based' in metrics is of type bool
    #                         metrics['User_based'] = bool(user_based)
    #                         metrics['k'] = k
    #                         metrics['DataSet'] = dataset
    #                         # Append the dictionary to the existing DataFrame
    #                         metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=[0])], ignore_index=True)
    #     #save the dataframe
    #     metrics_df.to_csv('/home/bbruno/all_here/python course/vinnie/data/cleaned_data/metrics_df_knn.csv', index=False)
    #     print(metrics_df)
    # ##### END OF KNN algorithms #####
    
    # ##### START OF SVD, SVD++, NMF algorithms #####
    # # Create an empty dataframe
    # metrics_df2 = pd.DataFrame(
    #     columns=['Model','Biased','RMSE','MAE','MSE', 'Precision', 'Recall', 'Accuracy', 'DataSet'])

    # # iterate over all datasets
    # for dataset in datasets:
    #     data = pd.read_csv(f'/home/bbruno/all_here/python course/vinnie/data/cleaned_data/{dataset}.csv')
        
    #     # Iterate over all the algorithms
    #     for algorithm_class in [SVD, NMF]:
    #         # Iterate over the biased algorithms
    #         for biased in [True, False]:
    #             algo = {'algorithm_class': algorithm_class, 'biased': biased}
    #             print(f"algorithm: {algo['algorithm_class']}")
    #             algorithms_instance = DynamicAlgo(**algo)
    #             algorithms_instance.create_reader(data)
    #             predictions = algorithms_instance.fit()
    #             algorithms_instance.get_accuracy(predictions)
            
    #             # metrics
    #             metrics = algorithms_instance.get_accuracy(predictions)
    #             metrics['Model'] = algo['algorithm_class'].__name__.split('.')[-1]
    #             metrics['Biased'] = biased
    #             metrics['DataSet'] = dataset
    #             # Append the dictionary to the existing DataFrame
    #             metrics_df2 = pd.concat([metrics_df2, pd.DataFrame(metrics, index=[0])], ignore_index=True)
            
    #     #iterate over the non-biased algorithms
    #     for algorithm_class in [SVDpp, SlopeOne, CoClustering]:
    #         biased = False
    #         algo = {'algorithm_class': algorithm_class, 'biased': biased}
    #         print(f"algorithm: {algo['algorithm_class']}")
    #         algorithms_instance = DynamicAlgo(**algo)
    #         algorithms_instance.create_reader(data)
    #         predictions = algorithms_instance.fit()
    #         algorithms_instance.get_accuracy(predictions)
        
    #         # metrics
    #         metrics = algorithms_instance.get_accuracy(predictions)
    #         metrics['Model'] = algo['algorithm_class'].__name__.split('.')[-1]
    #         metrics['Biased'] = biased
    #         metrics['DataSet'] = dataset
    #         # Append the dictionary to the existing DataFrame
    #         metrics_df2 = pd.concat([metrics_df2, pd.DataFrame(metrics, index=[0])], ignore_index=True)
    # #save the dataframe
    #     metrics_df2.to_csv('/home/bbruno/all_here/python course/vinnie/data/cleaned_data/metrics_df_svd.csv', index=False)
    #     print(metrics_df2)
    # ##### END OF SVD algorithms #####
    




















    # algorithms = [
    #     {'algorithm_class': KNNBasic, 'k': 5, 'sim_options': {'name': 'pearson_baseline','user_based': False}, 'bsl_options':{'method': 'sgd', 'learning_rate': 0.05, 'n_epochs':60, 'reg_u': 12 , 'reg_i': 5}},
    #     {'algorithm_class': KNNWithMeans, 'k': 5, 'sim_options': {'name': 'pearson_baseline','user_based': False}, 'bsl_options':{'method': 'sgd', 'learning_rate': 0.05, 'n_epochs':60, 'reg_u': 12 , 'reg_i': 5}},
    #     {'algorithm_class': KNNWithZScore, 'k': 5, 'sim_options': {'name': 'pearson_baseline','user_based': False}, 'bsl_options':{'method': 'sgd', 'learning_rate': 0.05, 'n_epochs':60, 'reg_u': 12 , 'reg_i': 5}},
    #     ]
    # for algo in algorithms:
    #     i = 1
    #     print(f"algorithm: {algo['algorithm_class']}")
    #     algorithms_instance = DynamicAlgo(**algo)
    #     algorithms_instance.create_reader(data)
    #     predictions = algorithms_instance.fit()
    #     algorithms_instance.get_accuracy(predictions)
    #     # metrics
    #     metrics = algorithms_instance.get_accuracy(predictions)
    #     metrics['Index'] = i
    #     metrics['Model'] = algo['algorithm_class']
    #     metrics['Metric'] = 
    #     metrics['Optimizer'] =
    #     metrics['User_based'] = 

    #     # Append the dictionary to the existing DataFrame
    #     metrics_df = metrics_df.append(metrics, ignore_index=True)

        

    # # algorithm architecture
    # knn_basic = DynamicAlgo(
    #     algorithm_class=KNNBasic, 
    #     k=5, 
    #     sim_options = {'name': 'pearson_baseline','user_based': False},
    #     bsl_options={'method': 'sgd', 'learning_rate': 0.05, 'n_epochs':60, 'reg_u': 12 , 'reg_i': 5})
    # # algorithm methods
    # knn_basic.create_reader(data)
    # predictions = knn_basic.fit()

    # knn_basic.get_neighbors_uid(user_id=5)
    # knn_basic.get_neighbors_iid(item_id=5)
    # knn_basic.estimated(u=5, i=5)
    # df_pred = knn_basic.inspect_predictions(predictions)
    # best_pred = df_pred.sort_values(by='err')[:10]
    # worst_pred = df_pred.sort_values(by='err')[-10:]
    # df_pred.head()

    # knn_basic.get_accuracy(predictions)
    # ##### END OF KNN BASIC #####

else:
    print("algorithms.py was imported!")
    
    

