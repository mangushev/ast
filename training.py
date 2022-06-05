import os
import sys
import argparse
import joblib
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_addons as tfa

from model import Generator, Discriminator
from evaluate import calculate_metrics

generator_attention = {'SOFTMAX': tf.nn.softmax, 'SPARSEMAX': tfa.activations.sparsemax}
generator_activation = {'SIGMOID': tf.nn.sigmoid, 'RELU': tf.nn.relu, 'SOFTPLUS': tf.math.softplus, 'NONE': None}

#tf.debugging.set_log_device_placement(True)

import numpy as np
np.set_printoptions(edgeitems=100000, linewidth=100000, precision=8, suppress=True)

FLAGS = None

def trans_parser(serialized_example):
  example = tf.io.parse_single_example(
    serialized_example,
    features={
      "condition": tf.io.FixedLenFeature([FLAGS.lookback_history, 1 + FLAGS.num_covariates], tf.float32),
      "label": tf.io.FixedLenFeature([FLAGS.estimate_length, 1 + FLAGS.num_covariates], tf.float32),
      "guid": tf.io.FixedLenFeature((), tf.int64)
    })

  example["guid"] = tf.cast(example["guid"], tf.int32)

  return example

def quantile_loss(y_real, y_fake):
  return 2*(tf.reduce_sum((y_fake - y_real)*(FLAGS.quantile*tf.cast(tf.math.greater(y_fake, y_real), tf.float32) - (1 - FLAGS.quantile)*tf.cast(tf.math.less_equal(y_fake, y_real), tf.float32)))/tf.reduce_sum(tf.math.abs(y_real)))

def metrics(actual, expense_estimate):

  performance = {}

  #tf.print ("A vs E: ", actual.shape, expense_estimate.shape, summarize=-1)
  #tf.print(actual, expense_estimate) 

  performance["mae"] = np.mean(np.absolute(actual - expense_estimate))
  performance["mbe"] = np.mean(actual - expense_estimate)
  #performance["rae"] = np.sum(np.absolute(actual - expense_estimate)) / np.sum(np.absolute(actual - np.mean(actual)))

  tf.print ("actial/zeros: ", actual.shape[0]*actual.shape[1], actual.shape[0]*actual.shape[1]-np.count_nonzero(actual))

  mask = np.ma.masked_equal(actual,0.0)
  e = np.ma.masked_where(np.ma.getmask(mask), expense_estimate)
  e = e.filled(fill_value=1)
  #e = e[e.mask == False]
  a = np.ma.masked_where(np.ma.getmask(mask), actual)
  a = a.filled(fill_value=1)
  #a = a[a.mask == False]

  #tf.print("what is a: ", a.data) 
  #tf.print(a.compressed().shape, e.compressed().shape) 
  #tf.print(a.compressed(), e.compressed()) 

  performance["mape"] = (np.mean(np.absolute((a - e) / a))) * 100

  #performance["smape"] = np.mean(np.absolute(actual - expense_estimate) / (np.absolute(actual) + np.absolute(expense_estimate))) * 100

  performance["smape"] = np.mean(np.absolute(a - e) / (np.absolute(a) + np.absolute(e))) * 100

  performance["mse"] = np.mean(np.square(actual - expense_estimate))
  performance["rmse"] = np.sqrt(np.mean(np.square(actual - expense_estimate)))
  #performance["rse"] = np.sum(np.square(actual - expense_estimate)) / np.sum(np.square(actual - np.mean(actual)))
  #performance["nrmse"] = np.sqrt(np.mean(np.square(actual - expense_estimate))) / np.std(expense_estimate)
  #performance["rrmse"] = np.sqrt(np.mean(np.square(actual - expense_estimate)) / np.sum(np.square(expense_estimate)))
  performance["qloss"] = quantile_loss(actual, expense_estimate)

  return performance

def evaluate():
  #   LEGEND:
  #account / weeks / features: dimentions
  #   a - accounts
  #   t - time dimension
  #   f - period transaction totals [type1, type2, type3, ..., total_debit, total_credit]
  trans_dataset = tf.data.TFRecordDataset(FLAGS.test_file)
  trans_dataset = trans_dataset.map(trans_parser)
  trans_dataset = trans_dataset.batch(FLAGS.batch_size, drop_remainder=False)

  generator = Generator(FLAGS.batch_size, 
		FLAGS.lookback_history,
		FLAGS.estimate_length,
		FLAGS.num_series,
		FLAGS.num_covariates,
                embedding_size=FLAGS.embedding_size,
		hidden_size=FLAGS.hidden_size,
                feedforward_size=FLAGS.feedforward_size,
		num_hidden_layers=FLAGS.num_hidden_layers,
		num_attention_heads=FLAGS.num_attention_heads,
		activation_fn=generator_activation[FLAGS.generator_activation],
		is_training=False)

  checkpoint_prefix = os.path.join(FLAGS.output_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator=generator)
  checkpoint.read(checkpoint_prefix).expect_partial()
  #status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir)).expect_partial()

  expense_estimate = np.empty([0, FLAGS.estimate_length], dtype=np.float32)
  actual = np.empty([0, FLAGS.estimate_length], dtype=np.float32)
  for trans_batch in trans_dataset:
    condition = trans_batch["condition"]
    covariates = trans_batch["label"][:, :, 1:]

    #tf.print(condition.shape)
    #tf.print(covariates)

    #sys.exit(0)

    while condition.shape[0] != FLAGS.batch_size:
      condition = tf.concat([condition, tf.zeros_like(condition[:1, :, :])], axis=0)
      covariates = tf.concat([covariates, tf.zeros_like(covariates[:1, :, :])], axis=0)

    estimate = generator(condition, covariates)

    expense_estimate = np.concatenate((expense_estimate, estimate[:trans_batch["label"].numpy().shape[0],:,0].numpy()), axis=0)
    actual = np.concatenate((actual, trans_batch["label"][:,:,0].numpy()), axis=0)

    if expense_estimate.shape[0] > 31450:
    #if expense_estimate.shape[0] > 64:
      break

  performance = metrics(actual, expense_estimate)

  with tf.io.gfile.GFile(FLAGS.output_file, "w") as writer:
    for m, v in performance.items():
      #tf.print("m/v", m, v)
      writer.write("{}: {:.6f} \n".format(m, v))

def evaluate_discriminator():
  #   LEGEND:
  #account / weeks / features: dimentions
  #   a - accounts
  #   t - time dimension
  #   f - period transaction totals [type1, type2, type3, ..., total_debit, total_credit]
  trans_dataset = tf.data.TFRecordDataset(FLAGS.test_file)
  trans_dataset = trans_dataset.map(trans_parser)
  trans_dataset = trans_dataset.batch(FLAGS.batch_size, drop_remainder=False)

  generator = Generator(FLAGS.batch_size, 
		FLAGS.lookback_history,
		FLAGS.estimate_length,
		FLAGS.num_series,
		FLAGS.num_covariates,
                embedding_size=FLAGS.embedding_size,
		hidden_size=FLAGS.hidden_size,
                feedforward_size=FLAGS.feedforward_size,
		num_hidden_layers=FLAGS.num_hidden_layers,
		num_attention_heads=FLAGS.num_attention_heads,
		activation_fn=generator_activation[FLAGS.generator_activation],
		is_training=False)

  discriminator = Discriminator(FLAGS.batch_size,
		FLAGS.estimate_length,
		hidden_size=FLAGS.estimate_length,
		activation_fn=tf.nn.leaky_relu,
		dropout_prob=0.0,
		is_training=False)

  checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
  status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir)).expect_partial()

  all_fake = np.empty([0], dtype=np.float32)
  all_real = np.empty([0], dtype=np.float32)
  for trans_batch in trans_dataset:
    condition = trans_batch["condition"]
    covariates_label = trans_batch["label"]

    while condition.shape[0] != FLAGS.batch_size:
      condition = tf.concat([condition, tf.zeros_like(condition[:1, :, :])], axis=0)
      covariates_label = tf.concat([covariates_label, tf.zeros_like(covariates_label[:1, :, :])], axis=0)

    estimate = generator(condition, covariates_label[:, :, 1:])

    discriminate_fake = discriminator(estimate[:, :, 0])
    discriminate_real = discriminator(covariates_label[:, :, 0])

    all_fake = np.concatenate((all_fake, discriminate_fake[:trans_batch["condition"].numpy().shape[0]].numpy()), axis=0)
    all_real = np.concatenate((all_real, discriminate_real[:trans_batch["condition"].numpy().shape[0]].numpy()), axis=0)

    performance = calculate_metrics(np.floor(np.concatenate((all_fake, all_real), axis=0)+0.5), np.concatenate((np.zeros_like(all_fake), np.ones_like(all_real)), axis=0))

    #performance = calculate_metrics(np.floor(all_fake+0.5), np.zeros_like(all_fake))
    performance = calculate_metrics(np.floor(all_real+0.5), np.ones_like(all_real))

    #if all_fake.shape[0] > 31450:
    if all_fake.shape[0] > 64:
      break

  print ("all_fake: ", all_fake)
  #print ("all_real: ", all_real)

  with tf.io.gfile.GFile(FLAGS.output_file, "w") as writer:
    for m, v in performance.items():
      #tf.print("m/v", m, v)
      writer.write("{}: {:.6f} \n".format(m, v))

def predict():
  #   LEGEND:
  #   a - accounts
  #   t - time dimension
  #   f - period transaction totals [type1, type2, type3, ..., total_debit, total_credit]
  trans_dataset = tf.data.TFRecordDataset(FLAGS.predict_file)
  trans_dataset = trans_dataset.map(trans_parser)
  trans_dataset = trans_dataset.batch(FLAGS.batch_size, drop_remainder=False)

  generator = Generator(FLAGS.batch_size, 
		FLAGS.lookback_history,
		FLAGS.estimate_length,
		FLAGS.num_series,
		FLAGS.num_covariates,
                embedding_size=FLAGS.embedding_size,
		hidden_size=FLAGS.hidden_size,
                feedforward_size=FLAGS.feedforward_size,
		num_hidden_layers=FLAGS.num_hidden_layers,
		num_attention_heads=FLAGS.num_attention_heads,
		activation_fn=generator_activation[FLAGS.generator_activation],
		is_training=False)

  checkpoint_prefix = os.path.join(FLAGS.output_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator=generator)
  checkpoint.read(checkpoint_prefix).expect_partial()
  #status = checkpoint.restore(tf.train.latest_checkpoint(FLAGS.output_dir)).expect_partial()

  scaler = joblib.load(FLAGS.predict_scaler_file)

  expense_estimate = np.empty([0, 1], dtype=np.float32)
  actual = np.empty([0, 1], dtype=np.float32)
  series = np.empty([0, 1], dtype=np.float32)
  guid = np.empty([0, 1], dtype=np.int32)

  for trans_batch in trans_dataset:
    condition = trans_batch["condition"]
    covariates= trans_batch["label"][:, :, 1:]

    while condition.shape[0] != FLAGS.batch_size:
      condition = tf.concat([condition, tf.zeros_like(condition[:1, :, :])], axis=0)
      covariates = tf.concat([covariates, tf.zeros_like(covariates[:1, :, :])], axis=0)

    estimate = generator(condition, covariates)

    expense_estimate = np.concatenate((expense_estimate, estimate[:trans_batch["label"].numpy().shape[0],23:24,0].numpy()), axis=0)
    actual = np.concatenate((actual, trans_batch["label"][:,23:24,0].numpy()), axis=0)
    series = np.concatenate((series, trans_batch["label"][:,:1,-1].numpy()), axis=0)
    guid = np.concatenate((guid, np.expand_dims(trans_batch["guid"].numpy(), axis=-1)), axis=0)

    #if expense_estimate.shape[0] > 31450:
    if expense_estimate.shape[0] > 64:
      break

#  for i in range(expense_estimate.shape[0]):
#    expense_estimate[i] = expense_estimate[i]*scaler[int(series[i, 0])][guid[i,0]]
#    actual[i] = actual[i]*scaler[int(series[i, 0])][guid[i,0]]

  with tf.io.gfile.GFile(FLAGS.output_file, "w") as writer:
    pct = 0
    n = 0
    for a, e in zip(actual[:,0], expense_estimate[:,0]):
      #writer.write("{:.2f} | {:.2f} \n".format(a, e))
      if a != 0.0:
        pct = pct + np.absolute((a - e) / a) * 100
        writer.write("{:.2f} | {:.2f} {:.0f}\n".format(a, e, np.absolute((a - e) / a) * 100))
      else:
        pct = pct + np.absolute(((a+0.1) - (e+0.1)) / (a+0.1)) * 100
        writer.write("{:.2f} | {:.2f} {}\n".format(a, e, '--'))
      n = n + 1
  print ("mape: ", pct/n) 

@tf.function
def train_loop(generator, discriminator, generator_optimizer, discriminator_optimizer, trans_dataset, checkpoint, checkpoint_prefix):
  #minimax problem: generator, looking for parameters to minimize that loss, for discriminator - parameters to maximize log_p_real & 1-log_p_fake(minus of it for Adam)
  for trans_batch in trans_dataset:
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
      start_time=time.time()
      condition = trans_batch["condition"]
      covariates = trans_batch["label"][:, :, 1:]

      y_fake = generator(condition, covariates)[:, :, 0]
      y_real = trans_batch["label"][:, :, 0]

      Lp = 2*tf.reduce_mean(tf.reduce_mean((y_fake - y_real)*(FLAGS.quantile*tf.cast(tf.math.greater(y_fake, y_real), tf.float32) - (1 - FLAGS.quantile)*tf.cast(tf.math.less_equal(y_fake, y_real), tf.float32))))

      #Lp = tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(y_fake, y_real)))

      #Lp = tf.reduce_mean(tf.math.abs(y_fake - y_real))

      regularizer = FLAGS.discriminator_lambda * tf.reduce_mean(tf.math.log(1 - tf.stop_gradient(discriminator(y_fake)) + 1e-15  ))

      generator_loss = Lp + regularizer

      generator_gradients = generator_tape.gradient(generator_loss, generator.variables)
      if (FLAGS.clip_gradients > 0):
        generator_gradients, _ = tf.clip_by_global_norm(generator_gradients, FLAGS.clip_gradients)

      generator_optimizer.apply_gradients(zip(generator_gradients, generator.variables))

      #it takes output sequense and returns true/false

      discriminator_fake = discriminator(tf.stop_gradient(y_fake))
      discriminator_real = discriminator(y_real)

      discriminator_loss = tf.reduce_mean(-tf.math.log(discriminator_real) - tf.math.log(1 - discriminator_fake +1e-15        ))

      discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.variables)
      if (FLAGS.clip_gradients > 0):
        discriminator_gradients, _ = tf.clip_by_global_norm(discriminator_gradients, FLAGS.clip_gradients)

      discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.variables))

      epoch = int((generator_optimizer.iterations * FLAGS.batch_size) / FLAGS.training_set_size)

      qloss = quantile_loss(y_real, y_fake)

      discriminator_accuracy = calculate_metrics(tf.floor(tf.concat([discriminator_fake, discriminator_real], axis=0)+0.5), tf.concat([tf.zeros_like(discriminator_fake), tf.ones_like(discriminator_real)], axis=0))['accuracy']

      tf.print("loss:", generator_optimizer.iterations, epoch, generator_loss, discriminator_loss, qloss, Lp, regularizer, discriminator_accuracy, generator_optimizer.lr(generator_optimizer.iterations), time.time() - start_time, output_stream=sys.stderr, summarize=-1)

      if generator_optimizer.iterations % FLAGS.save_batches == 0:
        #checkpoint.save(file_prefix=checkpoint_prefix)
        checkpoint.write(file_prefix=checkpoint_prefix)

      if (epoch > FLAGS.train_epochs):
        break

class WarmingSchedule(tf.optimizers.schedules.ExponentialDecay):
  def __init__(self, 
		warmup_steps,
		initial_learning_rate,
		minimal_learning_rate,
		decay_steps,
		decay_rate=0.99,
		staircase=False):
    super(WarmingSchedule, self).__init__(initial_learning_rate, decay_steps, decay_rate=decay_rate, staircase=staircase)

    self.warmup_steps = warmup_steps
    self.initial_learning_rate = initial_learning_rate
    self.minimal_learning_rate = minimal_learning_rate

  def __call__(self, step):
    rate = tf.case([(tf.equal(self.warmup_steps, 0), lambda: self.initial_learning_rate)], lambda: tf.minimum(self.initial_learning_rate*(1/self.warmup_steps)*tf.cast(step+1, tf.float32), super(WarmingSchedule, self).__call__(step)))
    return tf.case([(tf.less_equal(step, self.warmup_steps), lambda: rate)], lambda: tf.maximum(rate, self.minimal_learning_rate))

def train():
  #   LEGEND:
  #   B - batch
  #   t - time dimension
  #   f - feature dimention, main feature + covariates
  trans_dataset = tf.data.TFRecordDataset(FLAGS.train_file)
  trans_dataset = trans_dataset.repeat(-1)
  trans_dataset = trans_dataset.map(trans_parser)
  trans_dataset = trans_dataset.shuffle(100000, seed=0, reshuffle_each_iteration=True)
  trans_dataset = trans_dataset.batch(FLAGS.batch_size, drop_remainder=True)

  #0.0001 * 0.99 ^ (80000 / 1000) --> .000044
  #0.0001 * 0.99 ^ (40000 / 2000) --> .00008
  #0.0001 * 0.99 ^ (1000 / 5) --> .00001339
  #initial_learning_rate * decay_rate ^ (step / decay_steps)

  generator_learning_rate_fn = WarmingSchedule(FLAGS.warmup_steps, FLAGS.learning_rate, FLAGS.minimal_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)
  discriminator_learning_rate_fn = WarmingSchedule(FLAGS.warmup_steps, FLAGS.learning_rate, FLAGS.minimal_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)
  #generator_learning_rate_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.learning_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)
  #discriminator_learning_rate_fn = tf.optimizers.schedules.ExponentialDecay(FLAGS.learning_rate, FLAGS.decay_steps, decay_rate=0.99, staircase=False)

  generator_optimizer = tf.optimizers.Adam(generator_learning_rate_fn)
  discriminator_optimizer = tf.optimizers.Adam(discriminator_learning_rate_fn)

  generator = Generator(FLAGS.batch_size, 
		FLAGS.lookback_history,
		FLAGS.estimate_length,
		FLAGS.num_series,
		FLAGS.num_covariates,
                embedding_size=FLAGS.embedding_size,
                hidden_size=FLAGS.hidden_size,
		feedforward_size=FLAGS.feedforward_size,
		num_hidden_layers=FLAGS.num_hidden_layers,
		num_attention_heads=FLAGS.num_attention_heads,
		attention_fn=generator_attention[FLAGS.generator_attention],
		activation_fn=generator_activation[FLAGS.generator_activation],
		dropout_prob=FLAGS.dropout_prob,
		initializer_range=1.0,
		is_training=True)

  discriminator = Discriminator(FLAGS.batch_size,
		FLAGS.estimate_length,
		hidden_size=FLAGS.estimate_length,
		activation_fn=tf.nn.leaky_relu,
		dropout_prob=FLAGS.dropout_prob,
		initializer_range=1.0,
		is_training=True)

  checkpoint_prefix = os.path.join(FLAGS.output_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)
  import glob
  if len(glob.glob(checkpoint_prefix + '*')):
    checkpoint.read(checkpoint_prefix)

  train_loop(generator, discriminator, generator_optimizer, discriminator_optimizer, trans_dataset, checkpoint, checkpoint_prefix)

def main():  
  if FLAGS.action == 'TRAIN':
    train()
  elif FLAGS.action == 'EVALUATE':
    #evaluate()
    with tf.device('/cpu:0'):
      evaluate()
  elif FLAGS.action == 'EVALUATE_DISCRIMINATOR':
    with tf.device('/cpu:0'):
      evaluate_discriminator()
  elif FLAGS.action == 'PREDICT':
    #predict()
    with tf.device('/cpu:0'):
      predict()

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--output_dir', type=str, default='checkpoints',
            help='Model directrory in google storage.')
  parser.add_argument('--train_file', type=str, default='data/train.tfrecords',
            help='Train file location in google storage.')
  parser.add_argument('--test_file', type=str, default='data/test.tfrecords',
            help='Test file location in google storage.')
  parser.add_argument('--predict_file', type=str, default='data/predict.tfrecords',
            help='Predict file location in google storage.')
  parser.add_argument('--output_file', type=str, default='./output.csv',
            help='Prediction output.')
  parser.add_argument('--train_scaler_file', type=str, default='data/train_scaler.joblib',
            help='Scaling dollar amount.')
  parser.add_argument('--test_scaler_file', type=str, default='data/test_scaler.joblib',
            help='Scaling dollar amount.')
  parser.add_argument('--predict_scaler_file', type=str, default='data/predict_scaler.joblib',
            help='Scaling dollar amount.')
  parser.add_argument('--dropout_prob', type=float, default=0.1,
            help='This used for all dropouts.')
  parser.add_argument('--train_epochs', type=int, default=100,
            help='How many times to run scenarious.')
  parser.add_argument('--save_batches', type=int, default=1000,
            help='Save every N batches.')
  parser.add_argument('--num_series', type=int, default=369,
            help='Number of customers, each customer will have unique embedding.')
  parser.add_argument('--num_covariates', type=int, default=4,
            help='How many features total y and covariates.')
  parser.add_argument('--hidden_size', type=int, default=32,
            help='Transformer hidden size.')
  parser.add_argument('--embedding_size', type=int, default=20,
            help='Customer Index entry size in embedding table.')
  parser.add_argument('--feedforward_size', type=int, default=64,
            help='Last non-linearity layer in the transformer.')
  parser.add_argument('--num_hidden_layers', type=int, default=1,
            help='One self-attention block only.')
  parser.add_argument('--num_attention_heads', type=int, default=2,
            help='number of attention heads in transformer.')
  parser.add_argument('--lookback_history', type=int, default=12,
            help='How long is history used by estimator.')
  parser.add_argument('--estimate_length', type=int, default=2,
            help='Forecast length.')
  parser.add_argument('--quantile', type=float, default=0.5,
            help='Quantile value for loss calculation.')
  parser.add_argument('--discriminator_lambda', type=float, default=1.0,
            help='Hyperparameter to regularize generator loss.')
  parser.add_argument('--generator_attention', default='SOFTMAX', choices=['SOFTMAX','SPARSEMAX'],
            help='Used for transformer attentions.')
  parser.add_argument('--generator_activation', default='NONE', choices=['SIGMOID','RELU','SOFTPLUS','NONE'],
            help='To use any activation in generator output layer.')
  parser.add_argument('--learning_rate', type=float, default=1e-4,
            help='Optimizer initial learning rate.')
  parser.add_argument('--minimal_rate', type=float, default=5e-4,
            help='Optimizer minimal learning rate.')
  parser.add_argument('--decay_steps', type=int, default=50000,
            help='Exponential decay parameter.')
  parser.add_argument('--warmup_steps', type=int, default=50000,
            help='Learning rate grow from zero to initial rate during this time.')
  parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
  parser.add_argument('--batch_size', type=int, default=32,
            help='Batch size.')
  parser.add_argument('--clip_gradients', type=float, default=-1.0,
            help='Clip gradients to deal with explosive gradients.')
  parser.add_argument('--training_set_size', type=int, default=5805392,
            help='Batch size.')
  parser.add_argument('--action', default='PREDICT', choices=['TRAIN', 'EVALUATE', 'EVALUATE_DISCRIMINATOR', 'PREDICT'],
            help='An action to execure.')

  FLAGS, unparsed = parser.parse_known_args()

  tf.print ("Running with parameters: {}".format(FLAGS))

  main()
