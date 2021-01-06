import argparse
import librosa.display
from utils import *
import warnings


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
  
  


if __name__ == '__main__':
    main()
