from keras.models import model_from_json
from keras.optimizers import RMSprop
from features import *
import numpy as np
import pandas as pd


def predict(MFCC,MSS,poly,ZCR):

  # Load model
  model=Load_model()
  
  ans = model.predict([MFCC,MSS,poly,ZCR])
  ans = np.argmax(ans)

  return ans
  
def Load_model():
  # load json and create model
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("model.h5")

  # evaluate loaded model on test data
  loaded_model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=2e-5),metrics=['accuracy'])
  
  return loaded_model
  


def data_load_visual(samples,sample_rate):
  #Extract Feautures
  MFCC = mfcc_feature(samples , sample_rate)
  MSS = melspectrogram_feature(samples , sample_rate)
  poly = poly_feature(samples , sample_rate)
  ZCR = zero_crossing_rate_features(samples) 

  # sample graph
  sample_graph(samples, sample_rate)

  # Feaures graph
  MFCC_graph(MFCC)
  melspectrogram_graph(MSS)
  poly_graph(poly)
  zero_crossing_rate_graph(ZCR)
  
  return MFCC,MSS,poly,ZCR 
  

def Preprocessing(MFCC,MSS,poly,ZCR): 
  
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

  

