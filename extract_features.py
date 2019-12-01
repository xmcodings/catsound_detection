import numpy as np
max_pad_len = 775
import librosa



def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        #print(mfccs)
        pad_width = max_pad_len - mfccs.shape[1]
        #print(mfccs.shap)
        #print(pad_width)
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        #print(mfccs)
        print("parsing file: ", file_name)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs

