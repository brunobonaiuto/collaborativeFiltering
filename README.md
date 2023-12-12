Dude to the Bias observed during EDA on the data set, some 'Sampling Techniques' were applied:

* the full data set is called `merged_df.csv`:  it is the Original dataset.

* Undersampling the full data set we get:
    * `downsampled_df_random.csv` [by users]: it is taking a random sample from users with very large rates, such that assigning 20 as the maximun (it's not the best approach since basically the data was cutted)
    * `df_under.csv` [by rates]: it is taking a random sample from rates with too many users, such that the classes were balanced to some minimum in this case 175 samples. 
        * ( 19, 3) = 1 start      ------>     ( 19, 3) = 1 start
        * ( 87, 3) = 2 starts     ------>     ( 87, 3) = 2 starts
        * (346, 3) = 3 strats     ------>     (175, 3) = 3 starts   
        * (381, 3) = 4 starts     ------>     (175, 3) = 4 starts
        * (175, 3) = 5 starts     ------>     (175, 3) = 5 starts

* Oversampling the full data set we get:
    * `df_oversamling.csv` [by rates]: it is taking a random sample from rates with very few users, such that the classes were balanced to some maximum in this case 381 samples. 
        * ( 19, 3) = 1 start      ------>     (381, 3) = 1 start
        * ( 87, 3) = 2 starts     ------>     (381, 3) = 2 starts
        * (346, 3) = 3 strats     ------>     (381, 3) = 3 starts   
        * (381, 3) = 4 starts     ------>     (381, 3) = 4 starts
        * (175, 3) = 5 starts     ------>     (381, 3) = 5 starts
    * `upsampled_df_smote.csv` Synthetic Minority Oversampling Technique (SMOTE) [by rates]: new instances are synthesized from the existing data, by using `k nearest neighbor`` to select a random nearest neighbor, and a synthetic instance is created randomly in feature space.
        * ( 19, 3) = 1 start      ------>     (261, 3) = 1 start
        * ( 87, 3) = 2 starts     ------>     (59, 3) = 2 starts
        * (346, 3) = 3 strats     ------>     (252, 3) = 3 starts   
        * (381, 3) = 4 starts     ------>     (261, 3) = 4 starts
        * (175, 3) = 5 starts     ------>     (120, 3) = 5 starts

