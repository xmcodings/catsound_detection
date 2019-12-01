import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
import matplotlib.pyplot as plt
import pandas as pd
import librosa

num_rows = 50
num_columns = 620
num_channels = 1
max_pad_len = 775

def cat_analyze(filename):

    model = load_model('model/1202model/model.h5')
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    # Display model architecture summary 
    model.summary()
    label = pd.read_csv("label_to_text.csv")
    label_list = list(label['label'])
    
    #filename = 'test/test1.wav'
    
    
    try:
        audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        #print(mfccs)
        pad_width = max_pad_len - mfccs.shape[1]
        #print(mfccs.shap)
        #print(pad_width)
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        #print(mfccs)
        print("parsing file: ", filename)
    except Exception as e:
        print("Error encountered while parsing file: ", filename)
        return None 
         
    prediction_feature = mfccs
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)
    
    predicted_vector = model.predict_classes(prediction_feature)
    
    print("The predicted class is:", predicted_vector[0], '\n') 
    print("The predicted class is:", label_list[predicted_vector[0]], '\n')
    #print(predicted_class)
    #print("Accuracy:", predicted_class[1], '\n') 
    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    print(predicted_proba[predicted_vector[0]])
    print(predicted_proba[5])
    
    if(predicted_proba[predicted_vector[0]] == predicted_proba[5] and predicted_proba[5] > 0.4):
        print("great")
    elif(predicted_proba[predicted_vector[0]] == predicted_proba[5] and predicted_proba[5] > 0.3):
        print("good")
    
#check_accuracy.print_prediction_simple(filename, model) 

