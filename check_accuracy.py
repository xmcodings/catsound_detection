import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
import extract_features

le = LabelEncoder()
num_rows = 50
num_columns = 620
num_channels = 1

def print_prediction(file_name, model):
    prediction_feature = extract_features.extract_feature(file_name) 

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 
    #print(predicted_class)
    #print("Accuracy:", predicted_class[1], '\n') 
    

    predicted_proba_vector = model.predict_proba(prediction_feature) 
    
    predicted_proba = predicted_proba_vector[0]
    category = le.inverse_transform(np.array([1]))
    #print(category)
    print(predicted_proba[predicted_vector[0]])
    #print(predicted_proba(predicted_vector[0]))
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(i)
        #print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
        
        
        

def print_prediction_simple(file_name, model):
    prediction_feature = extract_features.extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
 
   # predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_vector, '\n') 
    #print(predicted_class)
    #print("Accuracy:", predicted_class[1], '\n') 
    

    predicted_proba_vector = model.predict_proba(prediction_feature) 
    
    predicted_proba = predicted_proba_vector[0]
    #print(model.)
    #category = le.inverse_transform(np.array([1]))
    #print(category)
    print(predicted_proba[predicted_vector[0]])
    #print(predicted_proba(predicted_vector[0]))
    #for i in range(len(predicted_proba)): 
        #category = le.inverse_transform(np.array([i]))
        
        #print(i)
        #print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
        