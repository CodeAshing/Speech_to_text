import numpy as np
import pandas as pd
import argparse
import librosa.display
from features import *
import warnings

def main():
  warnings.filterwarnings('ignore')
  
  parser = argparse.ArgumentParser(
        description='')
  parser.add_argument('--data_path', help='Path to folder containing documents ending in .mp3',
                      required=True)
  
  args = parser.parse_args()
  path = args.data_path

  sample_rate = 16000 


  # convert into number
  samples, sample_rate=librosa.load(path, sr=sample_rate)


  #Extract Feautures
  MFCC = mfcc_feature(samples , sample_rate)
  MSS = melspectrogram_feature(samples , sample_rate)
  poly = poly_feature(samples , sample_rate)
  ZCR = zero_crossing_rate_features(samples) 
  pitch = pitch_feature(samples , sample_rate)
  LPCC = lpcc_feature(samples , sample_rate)
  RPLP = RPLP_feature(samples , sample_rate)

  # sample graph
  sample_graph(samples, sample_rate)

  # Feaures graph
  MFCC_graph(MFCC)
  melspectrogram_graph(MSS)
  poly_graph(poly)
  zero_crossing_rate_graph(ZCR)
  lpcc_graph(LPCC)
  RPLP_graph(RPLP)
  pitch_graph(pitch)


if __name__ == '__main__':
    main()
