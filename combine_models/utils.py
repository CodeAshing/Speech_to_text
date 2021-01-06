import numpy as np
import pandas as pd
import warnings
import librosa
import joblib
from keras.models import load_model
from keras.models import model_from_json
from keras.optimizers import RMSprop
from features import *


def Main(sample):

  warnings.filterwarnings('ignore')
  
  
  #for Deep Learning with CNN
  sample_DL = process_data_for_DL_CNN(sample)
  
  #apply prediction
  pred_DL_CNN =predict_DL_CNN(sample_DL)  
  
  #for Deep Learning with LSTM
  sample_DL = process_data_for_DL_LSTM(sample)
  
  #apply prediction
  pred_DL_LSTM =predict_DL_LSTM(sample_DL)
  
  
  #for Machine Learning with random Forest
  
  #load, extract, and visualize the features 
  features = process_data_for_ML_Rf(sample)
  #prediction
  pred_ML_RF = predict_ML_RF(features)
  
  #for Machine Learning with KNN
  
  #load, extract, and visualize the features 
  features = process_data_for_ML_KNN(sample)
  #prediction
  pred_ML_KNN = predict_ML_KNN(features)
  
  #for Deep learning wih features
  
  MFCC,MSS,poly,ZCR = process_data_for_DL_FE(sample)
  #prediction
  pred_DL_FE = predict_DL_FE(MFCC,MSS,poly,ZCR)
  
  return pred_DL_CNN, pred_ML_RF, pred_DL_FE, pred_ML_KNN, pred_DL_LSTM	

# for Deep learning  
def predict_DL_LSTM(sample):
  # Load model
  model=load_model('DE_LSTM.h5')
  ans = model.predict(sample)
  ans = np.argmax(ans)

  return ans

def process_data_for_DL_LSTM(samples):
  sample_rate = 16000 
  max_len = 8000
  # convert into number
  samples = librosa.resample(samples, sample_rate, 8000)

  if(len(samples)== 8000) :
    samples = samples.reshape(-1,8000,1)
    return samples
  
  samples = samples[:max_len]
  samples = samples.reshape(-1,8000,1)
  return samples
  
  
# for Deep learning  
def predict_DL_CNN(sample):
  # Load model
  model=load_model('DL_CNN.hdf5')
  ans = model.predict(sample)
  ans = np.argmax(ans)

  return ans

def process_data_for_DL_CNN(samples):
  sample_rate = 16000 
  max_len = 8000
  # convert into number
  samples = librosa.resample(samples, sample_rate, 8000)

  if(len(samples)== 8000) :
    samples = samples.reshape(-1,8000,1)
    return samples
  
  samples = samples[:max_len]
  samples = samples.reshape(-1,8000,1)
  return samples
  
  
  
#for Machine Learning on KNN with Features
def predict_ML_KNN(features):

  loaded_model = joblib.load('model_KNN.sav')
  
  pred=loaded_model.predict(features.reshape(1,6000))
  
  return pred[0]

def process_data_for_ML_KNN(samples):

  sample_rate = 16000 
  #Extract Feautures
  MFCC = mfcc_feature(samples , sample_rate)
  MSS = melspectrogram_feature(samples , sample_rate)
  poly = poly_feature(samples , sample_rate)
  ZCR = zero_crossing_rate_features(samples) 

  # flatten an array
  MFCC = MFCC.flatten()
  MSS = MSS.flatten()
  poly = poly.flatten()
  ZCR = ZCR.flatten()
  
  #adding features into single array
  features = np.concatenate(( MFCC ,MSS, poly, ZCR))

  # padding and trimming
  max_len = 6000

  pad_width = max_len - features.shape[0]
  if pad_width > 0:
    features = np.pad(features, pad_width=((0, pad_width)), mode='constant')

  features = features[:max_len]

  return features
  
#for Machine Learning on random forest with Features
def predict_ML_RF(features):

  loaded_model = joblib.load('model_RF.sav')
  
  pred=loaded_model.predict(features.reshape(1,6000))
  
  return pred[0]

def process_data_for_ML_Rf(samples):

  sample_rate = 16000 
  #Extract Feautures
  MFCC = mfcc_feature(samples , sample_rate)
  MSS = melspectrogram_feature(samples , sample_rate)
  poly = poly_feature(samples , sample_rate)
  ZCR = zero_crossing_rate_features(samples) 

  # flatten an array
  MFCC = MFCC.flatten()
  MSS = MSS.flatten()
  poly = poly.flatten()
  ZCR = ZCR.flatten()
  
  #adding features into single array
  features = np.concatenate(( MFCC ,MSS, poly, ZCR))

  # padding and trimming
  max_len = 6000

  pad_width = max_len - features.shape[0]
  if pad_width > 0:
    features = np.pad(features, pad_width=((0, pad_width)), mode='constant')

  features = features[:max_len]

  return features
  
  
#for Deep Learningwih features


def predict_DL_FE(MFCC,MSS,poly,ZCR):

  # Load model
  model=Load_model_DL_FE()
  
  ans = model.predict([MFCC,MSS,poly,ZCR])
  ans = np.argmax(ans)

  return ans
  
def Load_model_DL_FE():
  # load json and create model
  json_file = open('DL_FE_model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("DL_FE_model.h5")

  # evaluate loaded model on test data
  loaded_model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=2e-5),metrics=['accuracy'])
  
  return loaded_model
  


def process_data_for_DL_FE(samples):

  sample_rate = 16000 
  #Extract Feautures
  MFCC = mfcc_feature(samples , sample_rate)
  MSS = melspectrogram_feature(samples , sample_rate)
  poly = poly_feature(samples , sample_rate)
  ZCR = zero_crossing_rate_features(samples) 

  max_len = 300
  #Normalizing
  MFCC = normalize_2d(MFCC)

  # zero-pad the mfccs features in order to have all compatible shapes for input of the CNN.
  # max_pad_len is the biggest number of audio frames   
  # obtained by extracting features from all the audio files.
  
  pad_width = max_len - MFCC.shape[1]
  if pad_width > 0:
    MFCC = np.pad(MFCC, pad_width=((0,0), (0, pad_width)), mode='constant')
  
  pad_width = max_len - MSS.shape[1]
  if pad_width > 0:
    MSS = np.pad(MSS, pad_width=((0,0), (0, pad_width)), mode='constant')
    
  pad_width = 1536 - poly.shape[1]
  if pad_width > 0:
    poly = np.pad(poly, pad_width=((0,0), (0, pad_width)), mode='constant')

  pad_width = 3072 - ZCR.shape[1]
  if pad_width > 0:
    ZCR = np.pad(ZCR, pad_width=((0,0), (0, pad_width)), mode='constant')


  #Trimming the array upto fix size

  MFCC = MFCC[:,:max_len]
  MSS = MSS[:,:max_len]
  poly = poly[:,:1536]
  ZCR = ZCR[:,:3072]
  
  #create and save data into DataFrames
  Featured_data = pd.DataFrame(columns=['MFCC', 'Mel-scaled-spectrogram', 'Poly','ZCR'])                
  Featured_data.loc[0] = [MFCC,MSS,poly,ZCR]
  
  
  MFCC = np.array(Featured_data.MFCC.tolist())           
  MSS = np.array(Featured_data['Mel-scaled-spectrogram'].tolist())
  poly = np.array(Featured_data.Poly.tolist())
  ZCR = np.array(Featured_data.ZCR.tolist())


  # Reshaping
  MFCC = MFCC.reshape(MFCC.shape[0],80,50,3)
  MSS = MSS.reshape(MSS.shape[0],128, 100,3)
  poly = poly.reshape(poly.shape[0],32, 32,3)
  ZCR = ZCR.reshape(ZCR.shape[0],32, 32,3)


  return MFCC,MSS,poly,ZCR 

  
def normalize_2d(v): 
  for i in range(v.shape[0]):
    norm = np.linalg.norm(v[i]) 
    if norm == 0: 
      v[i]= v[i] 
    else:
      v[i]= v[i] / norm
  return v



  

