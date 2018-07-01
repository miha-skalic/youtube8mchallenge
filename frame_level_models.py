# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

flags.DEFINE_bool("frame_shuffle", False,
                  "Set to true if you want to shuffle frames.")

from devel_models import *

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)


class GRUbidirect(models.BaseModel):
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   video_level_classifier_model=None,
                   lstm_size=None,
                   **unused_params):
    """Creates a model which uses 2 layer of GRUs to represent the video.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    # Use predefined and do not read from flags
    if not lstm_size:
        lstm_size = FLAGS.lstm_cells

    gru_fw = tf.contrib.rnn.GRUCell(lstm_size/2)
    gru_bw = tf.contrib.rnn.GRUCell(lstm_size/2)

    stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell([gru_fw])
    stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([gru_bw])

    outputs1, state1 = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw, stacked_lstm_bw,
                                                     model_input, sequence_length=num_frames, dtype=tf.float32)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(lstm_size)])

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, tf.concat(outputs1, axis=2),
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    if not video_level_classifier_model:
        video_level_classifier_model = FLAGS.video_level_classifier_model
    aggregated_model = getattr(video_level_models,
                               video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1],
        vocab_size=vocab_size,
        **unused_params)


def bidirect_process(pipe_input, n_cells, num_frames, scope_name):
    with tf.variable_scope(scope_name):
        gru_fw = tf.contrib.rnn.GRUCell(n_cells/2)
        gru_bw = tf.contrib.rnn.GRUCell(n_cells/2)

        stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell([gru_fw])
        stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([gru_bw])

        outputs1, state1 = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw, stacked_lstm_bw,
                                                           pipe_input, sequence_length=num_frames, dtype=tf.float32)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(n_cells)])

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, tf.concat(outputs1, axis=2),
                                           sequence_length=num_frames,
                                           dtype=tf.float32)
    return outputs, state


class GRUbidirect_branchedBN(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     is_training=True,
                     **unused_params):
        video_in = model_input[:, :, :1024]
        audio_in = model_input[:, :, 1024:]

        video_bn = slim.batch_norm(
            video_in,
            center=True,
            scale=True,
            is_training=is_training,
            scope="video_input_bn")
        audio_bn = slim.batch_norm(
            audio_in,
            center=True,
            scale=True,
            is_training=is_training,
            scope="audio_input_bn")

        # Transform input
        activation_video = slim.fully_connected(video_bn, 1024, activation_fn=None, biases_initializer=None)
        activation_audio = slim.fully_connected(audio_bn, 128, activation_fn=None, biases_initializer=None)

        activation_video = slim.batch_norm(
            activation_video,
            center=True,
            scale=True,
            is_training=is_training,
            scope="video_cluster_bn")
        activation_audio = slim.batch_norm(
            activation_audio,
            center=True,
            scale=True,
            is_training=is_training,
            scope="audio_cluster_bn")

        activation_video = tf.nn.relu6(activation_video)
        activation_audio = tf.nn.relu6(activation_audio)

        outputs_video, state_video = bidirect_process(activation_video, 1024, num_frames, scope_name="video")
        outputs_audio, state_audio = bidirect_process(activation_audio, 128, num_frames, scope_name="audio")

        state = tf.concat([state_video[-1], state_audio[-1]], axis=1)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            **unused_params)


class Lstmbidirect(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses 2 layer of LSTMs to represent the video.
    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    if FLAGS.frame_shuffle:
        model_input = utils.shuffle_frames(model_input, num_frames)
    lstm_size = FLAGS.lstm_cells

    lstm_fw = tf.contrib.rnn.BasicLSTMCell(lstm_size/2, forget_bias=1.0)
    lstm_bw = tf.contrib.rnn.BasicLSTMCell(lstm_size/2, forget_bias=1.0)

    stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell([lstm_fw])
    stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([lstm_bw])

    outputs1, state1 = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw, stacked_lstm_bw,
                                                     model_input, sequence_length=num_frames, dtype=tf.float32)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                ])

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, tf.concat(outputs1, axis=2),
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].c,
        vocab_size=vocab_size,
        **unused_params)
