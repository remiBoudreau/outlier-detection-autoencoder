# Filename: kdd_ae_utilities.py
# Dependencies: keras, numpy, pandas, sklearn, tensorflow
# Author: Jean-Michel Boudreau
# Date: May 16, 2019

# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import regularizers, optimizers
from keras.layers import Input, Dense#, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping


'''
Loads the kdd data set in csv format. File must be colocated with script.
Returns the kdd data as a pandas dataframe.
'''
def load_kdd_data():
	file_name = "kddcup.data"
	data = pd.read_csv(file_name)
    
	return data

'''
Processes the data (required as argument with type as pandas dataframe) for 
feeding into the autoencoder build with keras: 
    1. Scales the features in order to remove bias. 
    2. Randomly samples the data without replacement in order to create
       training, validation and testing subsets.
Returns validation, training and testing data subsets for the features
(labelled as 'X') as well as the validation and testing data target(labelled as
 'y').
'''
def process_data(data, test_pct=0.2, seed=42):
    # feature scaling to avoid bias in a single/few feature that are large in
    # magnitude as well as speed up convergence (when using any GD methods). 
    # See reference [2]
    scaler = MinMaxScaler()
    column_list = list(data)
    data[column_list] = scaler.fit_transform(data)
    # Randomly samples the data without replacement in order to create
    # training and testing subsets.
    data_train, data_test = train_test_split(data, 
                                             test_size=test_pct,
                                             random_state=seed)
    
    y_train = data_train['label'] # save the target
    X_train = data_train.drop(['label'], axis=1) # drop the target 
    
    y_test = data_test['label'] # save the target
    X_test = data_test.drop(['label'], axis=1) # drop the target 

    X_train = X_train.values # transform to ndarray
    X_test = X_test.values
    
    y_train = y_train.values
    y_test = y_test.values
    
    # Split the training data into validation and training datasets
    X_valid, X_train = X_train[:10000], X_train[10000:]
    y_valid = y_train[:10000]

    return X_valid, X_train, X_test, y_valid, y_test

'''
Creates an autoencoder that ...
i) encodes the 34 features into 17 and maps those 14 to 8 outputs
ii) decodes the 8 encoded outputs into 17 features and attempts reconstruction
    of the original data
'''
def get_autoencoder(input_dim):
    # parameters for standard "funnnel-like" network architecture
    encoding_dim = int(input_dim/2) # i.e. 17
    hidden_dim = int(encoding_dim / 2) # i.e. 8
    learning_rate = 1e-7
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    # encoder architecture
    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="elu", 
                activity_regularizer=regularizers.l1(learning_rate))(input_layer)
    # encoder = Dropout(0.4)(encoder) 
    encoder = Dense(hidden_dim, activation="elu")(encoder)
    # encoder = Dropout(0.4)(encoder) 
    
    # decoder architecture
    decoder = Dense(hidden_dim, activation='elu')(encoder)
    # encoder = Dropout(0.4)(decoder) 
    decoder = Dense(input_dim, activation='elu')(decoder)
    # decoder = Dropout(0.4)(decoder) 
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    # construct autoencoder
    autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer=sgd)
    return autoencoder

'''
Trains the autoencoder with the training data via mini-batch gradient descent
'''
def train_autoencoder(autoencoder, X_train, X_valid, n_epoch = 10, batch_size = 128):
    # checkpoint to retrieve parameters with minimal validation loss
    cp = ModelCheckpoint(filepath="kdd_ae.h5",
                     monitor='val_loss',
                     mode='min',
                     save_best_only=True,
                     verbose=0)
    # statement for early stopping callback to prevent overfitting and save time
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    # train autoencoder and save loss as a function of epoch
    history = autoencoder.fit(X_train, X_train,
                              epochs=n_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(X_valid, X_valid),
                              verbose=1,
                              callbacks=[cp, es]).history
    return history

'''
Ca. the mean squaerd error of each attempt with respect to all of its features
'''
def get_mse(autoencoder, X):
    X_pred = autoencoder.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)

    return mse

'''
Prints out statistics, most importantly, the mean and standard deviation, in
order to choose an appropriate threshold for the mean squared error that would
allow for classification of the attempt as malicious or not.
'''
def get_stats(autoencoder, X, v):
    mse = get_mse(autoencoder, X)
    if v == 0:
        mse = pd.DataFrame({'MSE_normal':mse[:]})
    elif v == 1:
        mse = pd.DataFrame({'MSE_malicious':mse[:]})
    print(mse.describe())
    
'''
Predicts the classification of each attempt based on the threshold specified
'''
def use_autoencoder_clf(autoencoder, X, threshold=0.05):
    mse = get_mse(autoencoder, X)
    nor_clf = mse < threshold
    y_pred = abs(nor_clf*1 - 1)
    
    return y_pred

'''
Prints out the confusion matrix of the results of the autoencoder with the pred
labels
'''
def print_stats(y_test, y_pred):
    accuracy_ae_clf = confusion_matrix(y_test, y_pred)
    print("Test data confusion matrix:")
    print(accuracy_ae_clf)
