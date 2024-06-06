"""Import the Packages"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Dense, Activation, LSTM, Dropout, Conv1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.utils import get_custom_objects
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


"""Change the Y Values"""
# seizure = 1
# no_seizure = 0

# raw_data['y'].replace(1, seizure, inplace=True)
# raw_data['y'].replace(2, no_seizure, inplace=True)
# raw_data['y'].replace(3, no_seizure, inplace=True)
# raw_data['y'].replace(4, no_seizure, inplace=True)
# raw_data['y'].replace(5, no_seizure, inplace=True)


"""Split Dataset into Training Set and Test Set"""
def splitting_data(perc,raw_data): 
    # Set for training the model
    data = raw_data.head(int(len(raw_data) * (perc / 100)))
    # Set for testing the model later to get the real accuracy
    test_set = raw_data.tail(int(len(raw_data) * ((100 - perc) / 100)))
    return data,test_set

"""Format the Data
    1. Split the data into Train and Test sets
    2. Get the data into the right shapes for training"""
def format_data(data):
    x_values = data.values[:, 1:-1]
    y_values = np.array(data['y'])
    y_values = np_utils.to_categorical(y_values)
    x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.2, random_state=1)
    x_train = x_train.reshape(-1, 178, 1)
    x_test = x_test.reshape(-1, 178, 1)
    return(x_values,y_values,x_train,y_train,x_test,y_test)
    # return(
    #     "X Train: {}\nX Test: {}\nY Train: {}\nY Test {}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape).split("\n"))


"""Visualize the Dataset"""
def Visualize_Dataset(x_values):
    # Set the size of the chart
    fig=plt.figure(figsize=(12, 8))
    # Plot data labeled 1 - Seizure
    plt.plot(x_values[1, :], label="1 - Epileptic")
    # Plot data labeled 2 - No Seizure
    plt.plot(x_values[7, :], label="2 - Not Epileptic")
    # Plot data labeled 3 - No Seizure
    plt.plot(x_values[12, :], label="3 - Not Epileptic")
    # Plot data labeled 4 - No Seizure
    plt.plot(x_values[0, :], label="4 - Not Epileptic")
    # Plot data labeled 5 - No Seizure
    plt.plot(x_values[2, :], label="5 - Not Epileptic")
    # Create a legend and output the graph
    plt.legend()
    #plt.savefig('figures/Seizure.png')
    plt.show()
    return fig

"""Model Training - Creating a Custom Activation Function - Swish"""
def custom_activation(x, beta=2):
    return K.sigmoid(beta * x) * x

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

"""Create a 3D CNN LSTM Model"""
def modle_creation():
    ## 1D CNN
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(45, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    ## LSTM
    model.add(LSTM(56, input_shape=(45, 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(56))
    model.add(Dropout(0.3))
    model.add(Dense(20))
    model.add(Activation(custom_activation, name='Swish'))
    model.add(Activation('tanh'))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    
    # Define the variables for training
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    # Output the model summary
    return model

"""Train the Model"""
def train_model(x_train,y_train,x_test,y_test,model):
    x=np.asarray((x_train[:, ::4] - x_train.mean()) / x_train.std()).astype('float32')
    y=np.asarray(y_train[:, 1:]).astype('float32')
    hist = model.fit(
        x = tf.convert_to_tensor(x, dtype = tf.float32),
        y = tf.convert_to_tensor(y, dtype = tf.float32),
        validation_data=(
            np.asarray((x_test[:, ::4] - x_test.mean()) / x_test.std()).astype('float32'),
            np.asarray(y_test[:, 1:]).astype('float32')
        ),
        epochs=100,
        batch_size=15,
        shuffle=True
    )
    return model,hist

# """Save the Model"""
# model_name = "1D_CNN_LSTM.h5"
# model.save(model_name)

"""Visualize Model's Accuracy Metrics"""
def Visualize_Accuracy(hist):
    # Plot Training Loss and Accuracy of the Model
    fig1=plt.figure(0)
    plt.plot(hist.history['loss'], 'green', label="1 - Training Loss")
    plt.plot(hist.history['accuracy'], 'red', label="2 - Training Accuracy")
    plt.legend()
    #plt.savefig('figures/Training Loss and Accuracy of the Model')
    plt.show()
    plt.close()
    # Plot Validation Loss and Accuracy of the Model
    fig2=plt.figure(0)
    plt.plot(hist.history['val_loss'], 'blue',label="1 - Validation Loss")
    plt.plot(hist.history['val_accuracy'], 'black', label="2 - Validation Accuracy")
    plt.legend()
    #plt.savefig('figures/Validation Loss and Accuracy of the Model')
    plt.show()
    return fig1,fig2

"""Split the Test Set"""
def split_test_set(test_set):
    x_test_values = test_set.values[:, 1:-1]
    y_test_values = np.array(test_set['y'])
    y_test_values = np_utils.to_categorical(y_test_values)
    x_test_values = x_test_values.reshape(-1, 178, 1)
    return (x_test_values,y_test_values)
    #print("x_test_values Shape: {}\ny_test_values Shape: {}".format(x_test_values.shape, y_test_values.shape))

"""Predict"""
def predict(model,x_test_values,y_test_values):
    predictions = model.predict(np.asarray((x_test_values[:, ::4] - x_test_values.mean()) / x_test_values.std()).astype(np.float32))
    """Format the Data
        Get the data into the right input shapes for the predictions"""
    y_pred = np.zeros((y_test_values.shape[0]))
    y_truth = np.ones((y_test_values.shape[0]))

    for i in range(y_test_values.shape[0]):
        y_pred[i] = np.argmax(predictions[i]) + 1
        y_truth[i] = np.argmax(y_test_values[i])

    for i in range(y_test_values.shape[0]):
        if y_truth[i] != 1:
            y_truth[i] = 0
        if y_pred[i] != 1:
            y_pred[i] = 0
    return y_truth,y_pred

"""Calculate the Accuracy"""
def accuracy(y_truth, y_pred):
    return(str((accuracy_score(y_truth, y_pred))*100), 
           (classification_report(y_truth, y_pred, output_dict=True)),
           (confusion_matrix(y_truth, y_pred)))