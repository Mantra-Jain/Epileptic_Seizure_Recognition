import streamlit as st
import pandas as pd
import oneD_CNN_LSTM
import numpy as np
import time
import json 
from streamlit_lottie import st_lottie

path1 = "/Users/mantrajain/Downloads/Animation.json"
with open(path1,"r") as file: 
    url1 = json.load(file) 

st.title("Epilepsy Detection")

"""Import the Dataset"""
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    raw_data = pd.read_csv(uploaded_file)
    st.write(raw_data)
    
"""Split Dataset into Training Set and Test Set"""
# Percentage to split by for training
perc = st.slider("Select the percentage of data for training", 0, 100, 90)
st.write("Trainig(%): ",perc)
st.write("Testing(%): ", (100-perc))
# Set for training the model
data, test_set=oneD_CNN_LSTM.splitting_data(perc,raw_data)

#formatting data
x_values,y_values,x_train,y_train,x_test,y_test=oneD_CNN_LSTM.format_data(data)
st.write("X_Train: ", x_train.shape)
st.write("X_Test: ", x_test.shape)
st.write("Y_Train: ", y_train.shape)
st.write("Y_Test: ", y_test.shape)

"""Visualize the Dataset"""
st.pyplot(oneD_CNN_LSTM.Visualize_Dataset(x_values))

"""Creating a Model"""
model=oneD_CNN_LSTM.modle_creation()
model.summary(print_fn=lambda x: st.text(x))

hist=None
"""Training the Model"""
if st.button("Start Training"):
    with st.spinner('Training the Model...'):
        st_lottie(url1, reverse=True, height=200, width=200, speed=1, loop=True, quality='high', key='Training')
        start_time = time.time()
        model,hist=oneD_CNN_LSTM.train_model(x_train,y_train,x_test,y_test,model)
        elapsed_time = time.time() - start_time
        st.write("time taken:",time.strftime("%M:%S", time.gmtime(elapsed_time)))
    st.success('Done!')

"""Visualize Model's Accuracy Metrics"""
training,validation=oneD_CNN_LSTM.Visualize_Accuracy(hist)
st.pyplot(training)
st.pyplot(validation)

"""Split the Test Set"""
x_test_values,y_test_values=oneD_CNN_LSTM.split_test_set(test_set)
st.write("x_test_values Shape:",x_test_values.shape)
st.write("y_test_values Shape:",y_test_values.shape)

"""Predict"""
y_truth,y_pred=oneD_CNN_LSTM.predict(model,x_test_values,y_test_values)

"""accuracy of model"""
accuracy,classification_report,cm=oneD_CNN_LSTM.accuracy(y_truth, y_pred)
st.write("Accuracy of Model: ", accuracy)
st.write("Confusion Matrix",cm)
st.write("Classification Report",classification_report)
st.dataframe(pd.DataFrame(classification_report))






