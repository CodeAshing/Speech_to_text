import numpy as np
import pandas as pd
import argparse
import librosa.display
import warnings
from collections import Counter 
from keras.utils import to_categorical 
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder
from features import *
from utils import *


def main():
  warnings.filterwarnings('ignore')
  
  global labels
  labels=['bed', 'cat', 'down', 'left', 'no', 'right', 'seven', 'stop', 'yes', 'up']
  parser = argparse.ArgumentParser(
        description='')
  parser.add_argument('--data_path', help='Path to folder containing documents ending in .mp3',
                      required=True)
  
  args = parser.parse_args()
  path = args.data_path


  #inialize labels
  sample_rate = 16000 

  # convert into number
  samples, sample_rate=librosa.load(path, sr=sample_rate)

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


  pred_DL_CNN, pred_ML_RF, pred_DL_FE, pred_ML_KNN, pred_DL_LSTM = Main(samples)
  
  #plot roc
  plot_ROC(pred_DL_CNN, pred_ML_RF, pred_DL_FE, pred_ML_KNN, pred_DL_LSTM)
  
  
  #gettings labels
  DL_ans_CNN=labels[pred_DL_CNN]
  ans_DL_LSTM=labels[pred_DL_LSTM]
  ML_ans_RF=labels[pred_ML_RF]
  ans_ML_KNN=labels[pred_ML_KNN]
  DL_FE_ans=labels[pred_DL_FE]
  
  #apply estimator
  #calling DL_ans_CNN and ML_ans_RF because you give priority to these two model because they have higher accuracy than other
  
  counter = Counter([DL_ans_CNN,DL_ans_CNN , ML_ans_RF,ML_ans_RF, DL_FE_ans, ans_ML_KNN,ans_DL_LSTM]) 
  most_occur = counter.most_common(1) 
   

  print(f'\n\n\nPredicions\n Machine Learning on random forest with Features: {ML_ans_RF} \n Machine Learning on KNN with Features: {ans_ML_KNN}\n Deep Learning (CNN) : {DL_ans_CNN}\n Deep Learning (LSTM) : {ans_DL_LSTM}\n Deep Learning with Feaures (CNN) : {DL_FE_ans}\n\n Prediction : {most_occur[0][0]}')


def plot_ROC(pred_DL_CNN, pred_ML_RF, pred_DL_FE, pred_ML_KNN, pred_DL_LSTM):
  
  le = LabelEncoder()
  encoded_labels = to_categorical(le.fit_transform(labels)) 
  y_expect =np.array([encoded_labels[pred_DL_CNN] for x in range(5)])
  y_predict = np.array([encoded_labels[x] for x in [pred_DL_CNN, pred_ML_RF, pred_DL_FE, pred_ML_KNN, pred_DL_LSTM]])
  
  
  # Compute ROC curve and ROC area for each class
  fpr = dict() 
  tpr = dict()
  roc_auc = dict()
  for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_expect[i], y_predict[i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_expect.ravel(), y_predict.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
 
                
  #indivitual plot
  plt.figure(num=None, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
  ax1 = plt.subplot2grid((2, 3), (0, 0))
  ax2 = plt.subplot2grid((2, 3), (0, 1))
  ax3 = plt.subplot2grid((2, 3), (0, 2))
  ax4 = plt.subplot2grid((2, 3), (1, 0))
  ax5 = plt.subplot2grid((2, 3), (1, 1))

  plt.subplots_adjust(wspace = 1.5, hspace = .5) 
  
  # Plot of a ROC curve for a specific class
  ax1.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
  ax1.plot([0, 1], [0, 1], 'k--')
  ax1.axis(xmin=0.0,xmax= 1.0)
  ax1.axis(ymin=0.0,ymax= 1.05)
  ax1.set_xlabel('False Positive Rate')
  ax1.set_ylabel('True Positive Rate')
  ax1.set_title('Receiver operating characteristic example')
  ax1.legend(loc="lower right")


  # Plot of a ROC curve for a specific class
  ax2.plot(fpr[1], tpr[1], label='ROC curve (area = %0.2f)' % roc_auc[1],color='g')
  ax2.plot([0, 1], [0, 1], 'k--')
  ax2.axis(xmin=0.0,xmax= 1.0)
  ax2.axis(ymin=0.0,ymax= 1.05)
  ax2.set_xlabel('False Positive Rate')
  ax2.set_ylabel('True Positive Rate')
  ax2.set_title('Receiver operating characteristic example')
  ax2.legend(loc="lower right")


  # Plot of a ROC curve for a specific class
  ax3.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2],color='y')
  ax3.plot([0, 1], [0, 1], 'k--')
  ax3.axis(xmin=0.0,xmax= 1.0)
  ax3.axis(ymin=0.0,ymax= 1.05)
  ax3.set_xlabel('False Positive Rate')
  ax3.set_ylabel('True Positive Rate')
  ax3.set_title('Receiver operating characteristic example')
  ax3.legend(loc="lower right")

  # Plot of a ROC curve for a specific class
  ax4.plot(fpr[3], tpr[3], label='ROC curve (area = %0.2f)' % roc_auc[3],color='r')
  ax4.plot([0, 1], [0, 1], 'k--')
  ax4.axis(xmin=0.0,xmax= 1.0)
  ax4.axis(ymin=0.0,ymax= 1.05)
  ax4.set_xlabel('False Positive Rate')
  ax4.set_ylabel('True Positive Rate')
  ax4.set_title('Receiver operating characteristic example')
  ax4.legend(loc="lower right")

  # Plot of a ROC curve for a specific class
  ax5.plot(fpr[4], tpr[4], label='ROC curve (area = %0.2f)' % roc_auc[4],color='C')
  ax5.plot([0, 1], [0, 1], 'k--')
  ax5.axis(xmin=0.0,xmax= 1.0)
  ax5.axis(ymin=0.0,ymax= 1.05)
  ax5.set_xlabel('False Positive Rate')
  ax5.set_ylabel('True Positive Rate')
  ax5.set_title('Receiver operating characteristic example')
  ax5.legend(loc="lower right")

  plt.show()
  
  
  # Plot ROC curve
  plt.figure(figsize=(10,10))
  plt.plot(fpr["micro"], tpr["micro"],
          label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]))
                                
  #combine ROC plot
  for i in range(5):
      plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Some extension of Receiver operating characteristic to multi-class')
  plt.legend(loc="lower right")
  plt.show()
  

  

if __name__ == '__main__':
    main()
