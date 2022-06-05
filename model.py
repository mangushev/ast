
#TODO:

import tensorflow as tf
import tensorflow_addons as tfa

import math
import six

import numpy as np

class Dense(tf.Module):
  def __init__(self, input_size, output_size, activation=None, stddev=1.0, name=''):
    super(Dense, self).__init__()
    self.w = tf.Variable(
      tf.random.truncated_normal([input_size, output_size], stddev=stddev), name=name + '_w')
    self.b = tf.Variable(tf.zeros([output_size]), name=name+'_b')
    self.activation = activation
    self.input_size = input_size
    self.output_size = output_size
  def __call__(self, x):
    input_shape = x.shape

    if len(input_shape) != 2 and len(input_shape) != 3:
      raise ValueError("input shape rank {} shuld be 2 or 3".format(len(input_shape)))

    if len(input_shape) == 3:
      #if self.input_size != input_shape[2]:
      #  raise ValueError("input size do not match {} {} {}".format(self.input_size, input_shape[2], input_shape))
      x = tf.reshape(x, [-1, self.input_size])
    else:
      x = x
      #if self.input_size != input_shape[1]:
      #  raise ValueError("input size do not match {} {}".format(self.input_size, input_shape[1]))

    y = tf.matmul(x, self.w) + self.b
    if (self.activation is not None):
      y = self.activation(y)

    if len(input_shape) == 3:
      return tf.reshape(y, [-1, input_shape[1], self.output_size])

    return y

class AttentionLayer(tf.Module):
  def __init__(self,
		batch_size,
		from_sequence_length,
		to_sequence_length,
		num_attention_heads,
		head_size,
		attention_mask=None,
		attention_fn=tf.nn.softmax,
		activation_fn=None,
		initializer_range=1.0):
    super(AttentionLayer, self).__init__()

    self.batch_size = batch_size
    self.from_sequence_length = from_sequence_length
    self.to_sequence_length = to_sequence_length
    self.num_attention_heads = num_attention_heads
    self.head_size = head_size
    self.attention_mask = attention_mask
    self.attention_fn = attention_fn

    self.query_layer = Dense(num_attention_heads * head_size,
		num_attention_heads * head_size,
		activation=activation_fn,
		stddev=initializer_range,
		name='query_layer')

    self.key_layer = Dense(num_attention_heads * head_size,
		num_attention_heads * head_size,
		activation=activation_fn,
		stddev=initializer_range,
		name='key_layer')

    self.value_layer = Dense(num_attention_heads * head_size,
		num_attention_heads * head_size,
		activation=activation_fn,
		stddev=initializer_range,
		name='value_layer')

  def __call__(self, to_sequence, from_sequence):

    query = self.query_layer(to_sequence)
    #(b, t, d) --> (b, t, h, dh)
    Q = tf.reshape(query, [self.batch_size, self.to_sequence_length, self.num_attention_heads, self.head_size])
    #(b, t, h, dh) --> (b, h, t, dh)
    Q = tf.transpose(Q, [0, 2, 1, 3])

    key = self.key_layer(from_sequence)
    #(b, f, d) --> (b, f, h, dh)
    K = tf.reshape(key, [self.batch_size, self.from_sequence_length, self.num_attention_heads, self.head_size])
    #(b, f, h, dh) --> (b, h, f, dh)
    K = tf.transpose(K, [0, 2, 1, 3])

    value = self.value_layer(from_sequence)
    #(b, f, d) --> (b, f, h, dh)
    V = tf.reshape(value, [self.batch_size, self.from_sequence_length, self.num_attention_heads, self.head_size])
    #(b, f, h, dh) --> (b, h, f, dh)
    V = tf.transpose(V, [0, 2, 1, 3])

    #(b, h, t, dh), (b, h, f, dh)T --> (b, h, t, f)
    output = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(self.num_attention_heads * tf.cast(self.head_size, tf.float32))

    if self.attention_mask is not None:
      #(b, t, f) --> (b, h, t, f)
      attention_mask = tf.tile(tf.expand_dims(self.attention_mask, axis=1), [1, self.num_attention_heads, 1, 1])
      attention_mask = -attention_mask * 10000

      #(b, h, t, f), (b, h, f, dh) --> (b, h, t, dh) 
      output = self.attention_fn(output + attention_mask)
      output = tf.matmul(output, V)
    else:
      output = tf.matmul(self.attention_fn(output), V)

    #(b, h, t, dh) --> (b, t, h, dh))
    output = tf.transpose(output, [0, 2, 1, 3])
    #(b, t, h, dh) --> (b, t, d)
    return tf.reshape(output, [self.batch_size, self.to_sequence_length, self.num_attention_heads * self.head_size])

def dropout(input, dropout_prob):
  if dropout_prob == 0.0:
    return input
  return tf.nn.dropout(input, dropout_prob)

class EncoderLayer(tf.Module):
  def __init__(self,
		batch_size,
		hidden_size,
		feedforward_size,
		sequence_length,
		num_attention_heads,
		head_size,
		attention_fn=tf.nn.softmax,
		activation_fn=None,
		feedforward_activation_fn=tf.nn.relu,
		dropout_prob=0.1,
		initializer_range=1.0):
    super(EncoderLayer, self).__init__()

    self.dropout_prob = dropout_prob

    self.self_attention_layer = AttentionLayer(
		batch_size,
		sequence_length,
		sequence_length,
		num_attention_heads,
		head_size,
		attention_mask=None,
		attention_fn=attention_fn,
		activation_fn=activation_fn,
		initializer_range=initializer_range)

    self.layer_normalization_1 = tfa.layers.GroupNormalization(groups=1, axis=-1)

    self.feedforward_layer = Dense(
		hidden_size,
		feedforward_size,
		activation=feedforward_activation_fn,
		stddev=initializer_range,
		name="feedforward_layer")

    self.layer_normalization_2 = tfa.layers.GroupNormalization(groups=1, axis=-1)

    self.project_back_layer = Dense(
		feedforward_size,
		hidden_size,
		activation=None,
		stddev=initializer_range,
		name="project_back_layer")

  def __call__(self, input):
    output = self.self_attention_layer(
		input,
		input)

    output = self.layer_normalization_1(output)

    attention_output = input + dropout(output, self.dropout_prob)

    output = self.feedforward_layer(attention_output)

    output = self.project_back_layer(output)

    output = self.layer_normalization_2(output)

    return attention_output + dropout(output, self.dropout_prob)

class DecoderLayer(tf.Module):
  def __init__(self,
		batch_size,
		hidden_size,
		feedforward_size,
		encoder_sequence_length,
		decoder_sequence_length,
		num_attention_heads,
		head_size,
		attention_mask,
		attention_fn=tf.nn.softmax,
		activation_fn=None,
		feedforward_activation_fn=tf.nn.relu,
		dropout_prob=0.1,
		initializer_range=1.0):
    super(DecoderLayer, self).__init__()

    self.dropout_prob = dropout_prob

    self.masked_attention_layer = AttentionLayer(
		batch_size,
                decoder_sequence_length,
                decoder_sequence_length,
		num_attention_heads,
		head_size,
		attention_mask=attention_mask,
		attention_fn=attention_fn,
		activation_fn=activation_fn,
		initializer_range=initializer_range)

    self.layer_normalization_1 = tfa.layers.GroupNormalization(groups=1, axis=-1)

    self.encoder_attention_layer = AttentionLayer(
		batch_size,
		encoder_sequence_length,
                decoder_sequence_length,
		num_attention_heads,
		head_size,
		attention_mask=None,
		attention_fn=attention_fn,
		activation_fn=activation_fn,
		initializer_range=initializer_range)

    self.layer_normalization_2 = tfa.layers.GroupNormalization(groups=1, axis=-1)

    self.feedforward_layer = Dense(
		hidden_size,
		feedforward_size,
		activation=feedforward_activation_fn,
		stddev=initializer_range,
		name="feedforward_layer")

    self.layer_normalization_3 = tfa.layers.GroupNormalization(groups=1, axis=-1)

    self.project_back_layer = Dense(
		feedforward_size,
		hidden_size,
		activation=None,
		stddev=initializer_range,
		name="project_back_layer")

  def __call__(self, input, encoder_attention):
    output = self.masked_attention_layer(
		input,
		input)

    output = self.layer_normalization_1(output)

    masked_attention_output = input + dropout(output, self.dropout_prob)

    output = self.encoder_attention_layer(
		masked_attention_output,
		encoder_attention)

    output = self.layer_normalization_2(output)

    attention_output = masked_attention_output + dropout(output, self.dropout_prob)

    output = self.feedforward_layer(attention_output)

    output = self.project_back_layer(output)

    output = self.layer_normalization_3(output)

    return attention_output + dropout(output, self.dropout_prob)

class Encoder(tf.Module):
  def __init__(self,
		batch_size,
		hidden_size,
		feedforward_size,
		sequence_length,
		num_layers,
		num_attention_heads,
		head_size,
		attention_fn=tf.nn.softmax,
		activation_fn=None,
		feedforward_activation_fn=tf.nn.relu,
		dropout_prob=0.1,
		initializer_range=1.0):
    super(Encoder, self).__init__()

    self.layers = []
    for _ in range(num_layers):
      self.layers.append(EncoderLayer(
		batch_size,
		hidden_size,
		feedforward_size,
		sequence_length,
		num_attention_heads,
		head_size,
		attention_fn,
		activation_fn,
		feedforward_activation_fn,
		dropout_prob,
		initializer_range))

  def __call__(self, input):
    output = input

    for layer in self.layers:
      output = layer(output)

    return output

class Decoder(tf.Module):
  def __init__(self,
		batch_size,
		hidden_size,
		feedforward_size,
		encoder_sequence_length,
		decoder_sequence_length,
		num_layers,
		num_attention_heads,
		head_size,
		masking=True,
		attention_fn=tf.nn.softmax,
		activation_fn=None,
		feedforward_activation_fn=tf.nn.relu,
		dropout_prob=0.1,
		initializer_range=1.0):
    super(Decoder, self).__init__()

    if masking:
      attention_mask = tf.tile(tf.expand_dims(tf.linalg.band_part(tf.ones([decoder_sequence_length, decoder_sequence_length], dtype=tf.float32), -1, 0), 0), [batch_size, 1, 1])
    else:
      attention_mask = None
    
    self.layers = []
    for _ in range(num_layers):
      self.layers.append(DecoderLayer(
		batch_size,
		hidden_size,
		feedforward_size,
		encoder_sequence_length,
		decoder_sequence_length,
		num_attention_heads,
		head_size,
		attention_mask,
		attention_fn,
		activation_fn,
		feedforward_activation_fn,
		dropout_prob,
		initializer_range))

  def __call__(self, input, encoder_attentions):
    output = input

    for layer in self.layers:
      output = layer(output, encoder_attentions)

    return output

class Transformer(tf.Module):
  def __init__(self,
		batch_size,
		hidden_size,
		feedforward_size,
		encoder_sequence_length,
		decoder_sequence_length,
		num_encoder_layers=2,
		num_decoder_layers=2,
		num_attention_heads=1,
		decoder_masking=True,
		attention_fn=tf.nn.softmax,
		activation_fn=None,
		feedforward_activation_fn=tf.nn.relu,
		dropout_prob=0.1,
		initializer_range=1.0):
    super(Transformer, self).__init__()

    head_size = tf.cast(hidden_size/num_attention_heads, dtype=tf.int32)

    self.encoder = Encoder(
		batch_size,
		hidden_size,
		feedforward_size,
		encoder_sequence_length,
		num_encoder_layers,
		num_attention_heads,
		head_size,
		attention_fn=attention_fn,
		activation_fn=activation_fn,
		feedforward_activation_fn=feedforward_activation_fn,
		dropout_prob=dropout_prob,
		initializer_range=initializer_range)
    self.decoder = Decoder(
		batch_size,
		hidden_size,
		feedforward_size,
		encoder_sequence_length,
		decoder_sequence_length,
		num_decoder_layers,
		num_attention_heads,
		head_size,
		masking=decoder_masking,
		attention_fn=attention_fn,
		activation_fn=activation_fn,
		feedforward_activation_fn=feedforward_activation_fn,
		dropout_prob=dropout_prob,
		initializer_range=initializer_range)
 
  def __call__(self, encoder_input, decoder_input):

    encoder_attentions = self.encoder(encoder_input)
    return self.decoder(decoder_input, encoder_attentions)

class Embedding(tf.Module):
  def __init__(self,
		batch_size,
		hidden_size,
		num_classes,
		dropout_prob=0.1,
		initializer_range=1.0):
    super(Embedding, self).__init__()

    self.batch_size = batch_size
    self.embedding_table = tf.Variable(tf.random.truncated_normal([num_classes, hidden_size], stddev=initializer_range), name='embedding_table')

  def __call__(self, index):
    #[I, d] --> [0, I, d]
    embedding_expanded = tf.expand_dims(self.embedding_table, 0)
    #[0, I, d] --> [B, I, d]
    #32, 370, 20
    embedding_expanded = tf.tile(embedding_expanded, [self.batch_size, 1, 1])

    #index: B, t
    #32, 168

    #[B, I, d] --> [B, t, d]
    return tf.gather(embedding_expanded, index, axis=1, batch_dims=1, name="embedding")

class Generator(tf.Module):
  #   B = batch size (number of sequences)
  #   n = lookback history
  #   o = output sequence size (1 day/week, 7 days/weeks)
  #   d - hidden size
  #   f = number of features
  #   e = number of estimated features
  def __init__(self, batch_size,
		lookback_history,
		estimate_length,
		num_series,
		num_covariates,
		embedding_size,
		hidden_size,
		feedforward_size,
		num_hidden_layers=2,
		num_attention_heads=2,
		attention_fn=tf.nn.softmax,
		activation_fn=tf.nn.relu,
		dropout_prob=0.1,
		initializer_range=1.0,
		is_training=False):
    super(Generator, self).__init__()

    def learnable_position_method(sequence_length, hidden_size):
      return tf.Variable(tf.random.truncated_normal([sequence_length, hidden_size], stddev=initializer_range), name='position_table')

    def sine_cosine_position_method(sequence_length, hidden_size):
      pos = tf.expand_dims(tf.range(sequence_length, delta=1, dtype=tf.float32), -1)
      position_table = tf.Variable(tf.zeros([sequence_length, hidden_size], dtype=tf.float32))
      position_table[:,0::2].assign(tf.math.sin(pos / tf.pow(10000.0, tf.expand_dims(tf.range(0, hidden_size, delta=2, dtype=tf.float32), 0) / hidden_size)))
      position_table[:,1::2].assign(tf.math.cos(pos / tf.pow(10000.0, tf.expand_dims(tf.range(1, hidden_size, delta=2, dtype=tf.float32), 0) / hidden_size)))
      return position_table

    if is_training == False:
      dropout_prob = 0.0   

    self.lookback_history = lookback_history
    self.estimate_length = estimate_length
    self.dropout_prob = dropout_prob

    self.encoder_alignment_layer = Dense(
		1 + num_covariates - 1 + embedding_size,
		hidden_size,
		stddev=initializer_range,
		name='encoder_alignment_layer')

    self.decoder_alignment_layer = Dense(
		num_covariates - 1 + embedding_size,
		hidden_size,
		stddev=initializer_range,
		name='decoder_alignment_layer')

    self.embedding = Embedding(
		batch_size=batch_size,
		hidden_size=embedding_size,
		num_classes=num_series,
		dropout_prob=dropout_prob,
		initializer_range=initializer_range)

    self.encoder_position_table = sine_cosine_position_method(lookback_history, hidden_size)
    self.decoder_position_table = sine_cosine_position_method(estimate_length, hidden_size)

    self.transformer_layer = Transformer(
		batch_size=batch_size,
		hidden_size=hidden_size,
		feedforward_size=feedforward_size,
		encoder_sequence_length=lookback_history,
		decoder_sequence_length=estimate_length,
		num_encoder_layers=num_hidden_layers,
		num_decoder_layers=num_hidden_layers,
		num_attention_heads=num_attention_heads,
		decoder_masking=False,
		attention_fn=attention_fn,
		activation_fn=None,
		feedforward_activation_fn=tf.nn.relu,
		dropout_prob=dropout_prob,
		initializer_range=initializer_range)

    self.output_layer = Dense(hidden_size, 
		1, 
		activation=activation_fn,
		stddev=initializer_range,
		name='output_layer')

    self.group_normalization_1 = tfa.layers.GroupNormalization(groups=1, axis=-1)
    self.group_normalization_2 = tfa.layers.GroupNormalization(groups=1, axis=-1)

  def __call__(self, condition, covariates):
    #(B, n, f) --> (B, n, d)
    encoder_input = tf.concat([condition[:, :, :-1], self.embedding(tf.cast(condition[:, :, -1], tf.int32))], axis=-1)
    encoder_input = self.encoder_alignment_layer(encoder_input) + tf.expand_dims(self.encoder_position_table, 0)
    encoder_input = self.group_normalization_1(encoder_input)
    encoder_input = dropout(encoder_input, self.dropout_prob)

    #(B, n, f) --> (B, n, d)
    decoder_input = tf.concat([covariates[:, :, :-1], self.embedding(tf.cast(covariates[:, :, -1], tf.int32))], axis=-1)
    decoder_input = self.decoder_alignment_layer(decoder_input) + tf.expand_dims(self.decoder_position_table, 0)
    decoder_input = self.group_normalization_2(decoder_input)
    decoder_input = dropout(decoder_input, self.dropout_prob)

    #(B, n, d) + (B, n, d) --> (B, n, 1)
    output = self.transformer_layer(encoder_input, decoder_input)

    return self.output_layer(output)

class Discriminator(tf.Module):
  #   B = batch size (number of sequences)
  #   n = lookback history
  #   o = output sequence size
  #   d - hidden size
  #   f = number of features
  #   e = number of estimated features
  def __init__(self, batch_size,
                     input_size,
                     hidden_size,
                     activation_fn=tf.nn.leaky_relu,
                     dropout_prob=0.1,
                     initializer_range=1.0,
                     is_training=False):
    super(Discriminator, self).__init__()

    if is_training == False:
      dropout_prob = 0.0   

    self.dropout_prob = dropout_prob

    self.layer_1 = Dense(
			input_size, 
    			hidden_size,
			activation=activation_fn,
            		stddev=initializer_range,
    			name='layer1')

    self.layer_2 = Dense(hidden_size, 
    			hidden_size,
			activation=activation_fn,
            		stddev=initializer_range,
    			name='layer_2')

    self.layer_3 = Dense(hidden_size, 
			1, 
			activation=tf.nn.sigmoid,
            		stddev=initializer_range,
			name='layer3')

    self.group_normalization_1 = tfa.layers.GroupNormalization(groups=1, axis=-1)
    self.group_normalization_2 = tfa.layers.GroupNormalization(groups=1, axis=-1)

  def __call__(self, y):
    output = self.layer_1(y)
    output = self.group_normalization_1(output)
    output = dropout(output, self.dropout_prob)
    output = self.layer_2(output)
    output = self.group_normalization_2(output)
    output = dropout(output, self.dropout_prob)
    #B, 1
    output = self.layer_3(output)
    output = tf.squeeze(output, axis=-1, name='factor_squeeze')
    return output
