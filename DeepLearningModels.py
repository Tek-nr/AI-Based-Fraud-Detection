import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from keras.models import Sequential
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Input, Dense, LSTM

##############################################################

optimizer = 'adam'
loss = 'binary_crossentropy'
metrics=[Precision(), Recall(), AUC()]

##############################################################

def reshape_train_and_test_sets_v1(X_train_scaled, X_test_scaled):
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    return X_train_reshaped, X_test_reshaped

def reshape_train_and_test_sets_v2(X_train_scaled, X_test_scaled):
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    return X_train_reshaped, X_test_reshaped

##############################################################

def y_pred_for_DLModels(model, X_test):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).flatten()
    return y_pred_binary

def evaluate_model(model, X_test_, y_test):
    test_loss, precision, recall, auc = model.evaluate(X_test_, y_test)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("*************************************************")
    print('F1 Score:', f1)
    print("*************************************************")
    return precision, recall, f1, auc


def fit_model(model, X_train_, y_train, validation_data=None, epochs=10, batch_size=128, shuffle=True):
    history = model.fit(X_train_, y_train, validation_split=0.2, validation_data=validation_data,
                        epochs=epochs, batch_size=batch_size, shuffle=shuffle)
    return history

##############################################################

def ANN_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # define the model
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=[X_train_scaled.shape[1]]),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # train the model
    fit_model(model, X_train_scaled, y_train)
    
    # make prediction
    y_pred = y_pred_for_DLModels(model, X_test_scaled)
    
    precision, recall, f1, auc = evaluate_model(model, X_test_scaled, y_test)
    row = "ANN", precision, recall, f1, auc
    return row

##############################################################    

def CNN_model(X_train_scaled, X_test_scaled, y_train, y_test):
    X_train_reshaped, X_test_reshaped = reshape_train_and_test_sets_v1(X_train_scaled, X_test_scaled) 
    
    model = keras.Sequential([
        layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    validation_data=(X_test_reshaped, y_test)
    
    fit_model(model, X_train_reshaped, y_train, validation_data)
    
    # make prediction
    y_pred = y_pred_for_DLModels(model, X_test_reshaped)
    
    precision, recall, f1, auc = evaluate_model(model, X_test_reshaped, y_test)
    row = "CNN", precision, recall, f1, auc
        
    return row

##############################################################

def autoencoders(X_train_scaled, X_test_scaled, y_train, y_test):
    
    # Define the dimensions of the input data
    input_dim = X_train_scaled.shape[1]

    # Define the Autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)

    # Create the Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compile the model
    autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    autoencoder.fit(X_train_scaled, X_train_scaled, epochs=10, batch_size=64, validation_data=(X_test_scaled, X_test_scaled))

    # Use the trained Autoencoder for anomaly detection
    reconstructed_data = autoencoder.predict(X_test_scaled)
    mse = np.mean(np.power(X_test_scaled - reconstructed_data, 2), axis=1)
    threshold = np.mean(mse) + np.std(mse)  # Define a threshold for anomaly detection

    # Classify instances as fraudulent or non-fraudulent based on the threshold
    y_pred = [1 if error > threshold else 0 for error in mse]
    
    # Calculate evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(y_test, mse)
    
    row = "Autoencoder", precision, recall, f1, auc
        
    return row

"""
def autoencoders(X_train_scaled, X_test_scaled, y_train, y_test):
    input_dim = X_train_scaled.shape[1]
    encoding_dim = 32

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation='relu')(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)
    
    model = Model(inputs=input_layer, outputs=decoder)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    validation_data=(X_test_scaled, X_test_scaled)
    
    fit_model(model, X_train_scaled, y_train, validation_data)
    
    # Obtain the reconstructed outputs
    reconstructed_X_test = autoencoder.predict(X_test_scaled)
    
    # Calculate the reconstruction error
    mse = np.mean(np.power(X_test_scaled - reconstructed_X_test, 2), axis=1)
    
    # Set a threshold for anomaly detection
    threshold = np.percentile(mse, 95)
    
    # Classify samples as normal or fraud based on the threshold
    y_pred = np.where(mse > threshold, 1, 0)
    
    precision, recall, f1, auc = evaluate_model(model, X_test_scaled, y_test)
    row = "Autoencoder", precision, recall, f1, auc
        
    return row
 """  

##############################################################
    
def RNN_model(X_train_scaled, X_test_scaled, y_train, y_test):
    
    X_train_reshaped, X_test_reshaped = reshape_train_and_test_sets_v2(X_train_scaled, X_test_scaled) 
    
    # Define the RNN model
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, X_train_scaled.shape[1]), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the RNN model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    validation_data=(X_test_reshaped, y_test)
    
    fit_model(model, X_train_reshaped, y_train, validation_data)
    
    y_pred = y_pred_for_DLModels(model, X_test_reshaped)
    
    precision, recall, f1, auc = evaluate_model(model, X_test_reshaped, y_test)
    row = "RNN", precision, recall, f1, auc
        
    return row

##############################################################

def LSTM_model(X_train_scaled, X_test_scaled, y_train, y_test):
    
    X_train_reshaped, X_test_reshaped = reshape_train_and_test_sets_v2(X_train_scaled, X_test_scaled) 
    
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, X_train_scaled.shape[1]), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    validation_data = (X_test_reshaped, y_test)
    
    fit_model(model, X_train_reshaped, y_train, validation_data)
    
    y_pred = y_pred_for_DLModels(model, X_test_reshaped)
    
    precision, recall, f1, auc = evaluate_model(model, X_test_reshaped, y_test)
    row = "LSTM", precision, recall, f1, auc
        
    return row