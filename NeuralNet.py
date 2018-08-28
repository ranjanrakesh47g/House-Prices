# Import packages
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import keras.backend as K
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt


# Hyperparameters/Parameters
Model = [276, 15000, 1000, 300, 1]
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
nb_epochs = 200
batch_size = 128


# Step I. Load data
os.chdir('/home/rakesh/House Prices')
X_train = pd.read_csv('X_train.csv')
#X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
y_train = pd.read_csv('y_train.csv', header=None).iloc[:, 0].rename('SalePrice')
#y_train = np.loadtxt('y_train.csv', delimiter=',')
X_test = pd.read_csv('X_test.csv')


# Step II. Build the model and perform parameter tuning

# Loss function : r2_score
def r2_score(y_true, y_pred):
    y_true_mean = K.mean(y_true)
    SS_tot = K.sum(K.square(y_true - y_true_mean))
    SS_reg = K.sum(K.square(y_true - y_pred))
    return (1 - SS_reg*1.0/SS_tot)

def loss_r2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return K.abs(1 - r2)

# Model
def create_model():
    model = Sequential()
    model.add(Dense(Model[1], input_dim=Model[0], activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(Model[2]))
    
    model.compile(loss=loss_r2, optimizer=adam, metrics=[r2_score])
    return model

model = Sequential()
model.add(Dense(Model[1], input_dim=Model[0], activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(Model[2], activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(Model[3], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Model[4]))

model.compile(loss=loss_r2, optimizer=adam, metrics=[r2_score])
history = model.fit(X_train.values, y_train.values, validation_split=0.1,
                    nb_epoch=nb_epochs, batch_size=batch_size)
                    
model = KerasRegressor(build_fn=create_model, nb_epoch=nb_epochs, batch_size=batch_size)
cv_r2_score = cross_val_score(model, X_train.values, y_train, cv=10, scoring='r2')
print('CV_coef_of_det: Mean-%0.6f | Std-%0.6f | Min-%0.6f | Max-%0.6f' %(np.mean(cv_r2_score),
        np.std(cv_r2_score), np.min(cv_r2_score), np.max(cv_r2_score)))



