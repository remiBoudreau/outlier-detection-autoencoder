# Filename: kdd_ae.py
# Dependencies: kdd_ae_utilities, keras, numpy, pandas, sklearn, tensorflow
# Author: Jean-Michel Boudreau
# Date: May 16, 2019

'''
Loads KDD data set (must be colocated in directory with this script) and 
trains an autoencoder to predict whether a given log-in attempt is malicious
(labelled as "1") or not (labelled as "0").
'''

# import libraries
from kdd_ae_utilities import load_kdd_data, process_data, get_autoencoder, train_autoencoder, get_stats, use_autoencoder_clf, print_stats

# load the kddcup data set
print('Reading kdd data file.')
kdd_data = load_kdd_data()

# process the data and split into feature and target training/testing subsets
print('Processing kdd data file.')
X_valid, X_train, X_test, y_valid, y_test = process_data(kdd_data)

# define autoencoder architecture
print('Building network.')
input_dim = X_train.shape[1] # num of features, 34
autoencoder = get_autoencoder(input_dim)

# train autoencoder
print('Training network.')
train_autoencoder(autoencoder, X_train, X_valid)

# split validation data into normal and malicious attempt data subsets
X_valid_nor = X_valid[y_valid == 0]
X_valid_mal = X_valid[y_valid == 1]
# get the mean and std of each subset
get_stats(autoencoder, X_valid_nor, 0)
get_stats(autoencoder, X_valid_mal, 1)
# threshold of 0.05 chosen for ad-hoc solution
print("Setting threshold to 0.05.")
threshold = 0.05

# use autoencoder to classify attempts based on mean MSE on test set
y_pred = use_autoencoder_clf(autoencoder, X_test)

# print performance measures of autoencoder binary classifier in confusion 
# matrix
print_stats(y_test, y_pred)
