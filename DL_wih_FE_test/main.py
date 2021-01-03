import numpy as np
import pandas as pd
import argparse
import librosa.display
from features import *
import warnings
from keras.models import model_from_json
from keras.optimizers import Adam, RMSprop


def main():
  warnings.filterwarnings('ignore')
  
  parser = argparse.ArgumentParser(
        description='')
  parser.add_argument('--data_path', help='Path to folder containing documents ending in .mp3',
                      required=True)
  
  args = parser.parse_args()
  path = args.data_path           

  #inialize labels
  labels = ['bed', 'cat', 'down', 'left', 'no', 'right', 'seven', 'stop', 'yes', 'up']

  
  sample_rate = 16000 

  # convert into number
  samples, sample_rate=librosa.load(path, sr=sample_rate)
  
  #load, extract, and visualize the features 
  MFCC,MSS,poly,ZCR = data_load_visual(samples,sample_rate)
  
  MFCC,MSS,poly,ZCR = Preprocessing(MFCC,MSS,poly,ZCR)
  
  #prediction
  pred = predict(MFCC,MSS,poly,ZCR)
  ans=labels[pred]
  
  print(f'\n\nPrediction: {ans}')
  
  
def predict(MFCC,MSS,poly,ZCR):

  # Load model
  model=Load_model()
  
  ans = model.evaluate((MFCC,MSS,poly,ZCR))
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
  #sample_graph(samples, sample_rate)

  # Feaures graph
 # MFCC_graph(MFCC)
  #melspectrogram_graph(MSS)
  #poly_graph(poly)
 # zero_crossing_rate_graph(ZCR)
  
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
  
  # Reshaping
  MFCC = MFCC.reshape(80,50,3)
  MSS = MSS.reshape(128, 100,3)
  poly = poly.reshape(32, 32,3)
  ZCR = ZCR.reshape(32, 32,3)


  return MFCC,MSS,poly,ZCR 

  



if __name__ == '__main__':
    main()
