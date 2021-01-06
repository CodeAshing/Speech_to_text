import numpy as np
import pandas as pd
import argparse
from keras.models import load_model
import warnings
import librosa

def main():
  warnings.filterwarnings('ignore')
  
  parser = argparse.ArgumentParser(
        description='')
  parser.add_argument('--data_path', help='Path to folder containing documents ending in .mp3',
                      required=True)
  
  args = parser.parse_args()
  path = args.data_path
  
  #inialize labels
  labels=['bed', 'cat', 'down', 'left', 'no', 'right', 'seven', 'stop', 'yes', 'up']
  #load and process data
  sample =load_data(path)
  
  #apply prediction
  pred =predict(sample)
  
  ans=labels[pred]
  
  print(f'\n\nprediction: {ans}')

  
def predict(sample):
  # Load model
  model=load_model('model.h5')
  ans = model.predict(sample)
  ans = np.argmax(ans)

  return ans

def load_data(path):
  sample_rate = 16000 
  max_len = 8000
  # convert into number
  samples, sample_rate=librosa.load(path, sr=sample_rate)
  samples = librosa.resample(samples, sample_rate, 8000)

  if(len(samples)== 8000) :
    samples = samples.reshape(-1,8000,1)
    return samples
  
  samples = samples[:max_len]
  samples = samples.reshape(-1,8000,1)
  return samples
  
  
if __name__ == '__main__':
    main()
