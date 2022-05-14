
#TODO:
#dividing by average, maybe collect all data from start and standardize?
#exclude outliers from training data?
#should I take date ranges when picking train and test instead of random?

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
import sys
import random
import logging
import datetime
import joblib
import heapq
from sklearn.preprocessing import MinMaxScaler, scale
from utils import Scaler

FLAGS = None

np.set_printoptions(edgeitems=1000, linewidth=10000, precision=2, suppress=True)

logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.removeHandler(logger.handlers[0])
logger.propagate = False

def data_example(condition, label, window_id):
  record = {
    'condition': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(condition, [-1]))),
    'label': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(label, [-1]))),
    'guid': tf.train.Feature(int64_list=tf.train.Int64List(value=[window_id]))
  }

  return tf.train.Example(features=tf.train.Features(feature=record))

def create_records(file_path, output_file, scaler_file, star_date, end_date, leading_zeros):
  account = pd.read_csv(file_path, sep=';', parse_dates=[0], thousands='.', decimal=',')
  account.rename(columns={'Unnamed: 0':'times'}, inplace=True)
  account.set_index(['times'],inplace=True)

  #time acc1 acc2
  #01:00 33   0   <--- 
  #02:00 22   0
  #03:00 44   44
  #04:00 55   11 

  #t, a
  account_grp = account[(account.index >= star_date) & (account.index < end_date)].resample('1H', axis=0, closed='right', label='left').sum()
  #account_grp = account.resample('1H', axis=0, closed='right', label='left').sum()
  #account_grp = account.resample('1H', axis=0, closed='left', label='left').sum()

  #(t, f)
  covariates = np.concatenate((
    np.expand_dims(scale(account_grp.index.dayofweek.values), axis=-1), 
    np.expand_dims(scale(account_grp.index.hour.values), axis=-1)), axis=-1).astype(np.float32)

  #t, a
  data = account_grp.to_numpy()

  #print (data[:, 222])

  #sys.exit(0)

  start_data = np.zeros_like(data[0, :], dtype=np.int)

  if leading_zeros:
    start_data = (data!=0).argmax(axis=0)

  #print ((np.flip(data, axis=0)!=0).argmax(axis=0))
  #print (data.shape[0] - (np.flip(data, axis=0)!=0).argmax(axis=0) - 1)

  #print ((data==0).all(axis=0))

  #print (np.count_nonzero(data, axis=0))
  #print (data.shape[0]-np.count_nonzero(data, axis=0))
  #print (data.shape)

  #sys.exit(0)
  
  #a, t, 1
  data = np.expand_dims(np.transpose(data, (1,0)), axis=-1)

  data = np.float32(data)

  covariates = np.concatenate((
    np.expand_dims(scale(account_grp.index.dayofweek.values), axis=-1), 
    np.expand_dims(scale(account_grp.index.hour.values), axis=-1)), axis=-1).astype(np.float32)

  #print (account_grp.index.dayofweek.values[:100])
  #print (scale(account_grp.index.dayofweek.values[:100]))

  #sys.exit(0)

  examples = []

  norm_values = []

  for i, a in enumerate(data):
    age = np.expand_dims(np.concatenate((np.zeros(start_data[i]), scale(np.arange(a.shape[0] - start_data[i]))), axis=-1), axis=-1)

    repeated_i = np.expand_dims(np.repeat(i, a.shape[0]), axis=-1)

    #t, f
    series_covariates = np.float32(np.concatenate((age, covariates, repeated_i), axis=1))

    serie_norm_values = []
    w = 0
    for h in range(start_data[i], a.shape[0]-(FLAGS.lookback_history+FLAGS.estimate_length+start_data[i]), FLAGS.estimate_length):

      #print (h, start_data[i], a.shape[0], FLAGS.lookback_history+FLAGS.estimate_length+start_data[i], a.shape[0]-(FLAGS.lookback_history+FLAGS.estimate_length+start_data[i]))

      window_data = a[h:h+FLAGS.lookback_history+FLAGS.estimate_length]
      window_covariates = series_covariates[h:h+FLAGS.lookback_history+FLAGS.estimate_length]

      nonzeros = np.count_nonzero(window_data)
   
      #print (a.shape)

      if nonzeros == 0:
        x = np.zeros_like(window_data)
        serie_norm_values.append(1)
      else:
        norm_value = (np.sum(window_data)/nonzeros)+1
        serie_norm_values.append(norm_value)

        x = window_data/norm_value

        x = np.concatenate((np.zeros((1,1)), x[:-1]), axis=0)

      x = np.float32(np.concatenate((x, window_covariates), axis=1))

      example = data_example(x[:FLAGS.lookback_history, :], x[FLAGS.lookback_history:FLAGS.lookback_history+FLAGS.estimate_length, :], w)

      examples.append(example)

      w = w + 1

    norm_values.append(serie_norm_values)

  random.shuffle(examples)

  with tf.io.TFRecordWriter(output_file) as writer:
    for example in examples:
      writer.write(example.SerializeToString())

  joblib.dump(norm_values, scaler_file)

  logger.info ("data shape: {}".format(data.shape))
  logger.info ("covariates shape: {}".format(covariates.shape))
  logger.info ("set sizes: {}".format(len(examples)))

def main():
  create_records(FLAGS.file_path, FLAGS.train_tfrecords_file, FLAGS.train_scaler_file, FLAGS.train_start_date, FLAGS.train_end_date, True)
  create_records(FLAGS.file_path, FLAGS.test_tfrecords_file, FLAGS.test_scaler_file, FLAGS.test_start_date, FLAGS.test_end_date, False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
          help='Enable excessive variables screen outputs.')
  parser.add_argument('--file_path', type=str, default='data/LD2011_2014.txt',
          help='Data file.')
  parser.add_argument('--train_tfrecords_file', type=str, default='data/train.tfrecords',
          help='train sales tfrecords output file')
  parser.add_argument('--test_tfrecords_file', type=str, default='data/test.tfrecords',
          help='test sales tfrecords output file')
  parser.add_argument('--train_scaler_file', type=str, default='data/train_scaler.joblib',
          help='Scaling dollar amount.')
  parser.add_argument('--test_scaler_file', type=str, default='data/test_scaler.joblib',
          help='Scaling dollar amount.')
  parser.add_argument('--lookback_history', type=int, default=168,
          help='How long is history used by estimator.')
  parser.add_argument('--estimate_length', type=int, default=24,
          help='Prediction range. This range should have covariates available.')
  parser.add_argument('--train_start_date', type=str, default='2011-01-01',
          help='Train start date inclusive')
  parser.add_argument('--train_end_date', type=str, default='2014-10-01',
          help='Train end date inclusive')
  parser.add_argument('--test_start_date', type=str, default='2014-10-01',
          help='Test start date inclusive')
  parser.add_argument('--test_end_date', type=str, default='2015-01-01',
          help='Test end date inclusive')

  FLAGS, unparsed = parser.parse_known_args()
  logger.setLevel(FLAGS.logging)
  logger.info ("Running with parameters: {}".format(FLAGS))

  main()
