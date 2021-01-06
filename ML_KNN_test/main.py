import numpy as np
import pandas as pd
import argparse
import librosa.display
from features import *
import warnings
import joblib

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
  features = data_load_visual(samples,sample_rate)
  
  #prediction
  pred = predict(features)
  ans=labels[pred]
  
  print(f'\n\nPrediction: {ans}')
  
  
def predict(features):

  loaded_model = joblib.load('model.sav')
  
  pred=loaded_model.predict(features.reshape(1,6000))
  
  return pred[0]

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
  
  features = Preprocessing(MFCC,MSS,poly,ZCR)
  
  return features
  

def Preprocessing(MFCC,MSS,poly,ZCR):
  # flatten an array
  MFCC = MFCC.flatten()
  MSS = MSS.flatten()
  poly = poly.flatten()
  ZCR = ZCR.flatten()
  
  
  # normalizing
  # MFCC = normalize(MFCC)

  #adding features into single array
  features = np.concatenate(( MFCC ,MSS, poly, ZCR))

  # padding and trimming
  max_len = 6000

  pad_width = max_len - features.shape[0]
  if pad_width > 0:
    features = np.pad(features, pad_width=((0, pad_width)), mode='constant')

  features = features[:max_len]

  return features

  



if __name__ == '__main__':
    main()
