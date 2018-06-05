"""
Models used for development.
"""

import math

import models
import video_level_models
import tensorflow as tf

import tensorflow.contrib.slim as slim
import model_utils as utils
from tensorflow import flags
FLAGS = flags.FLAGS


flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")
flags.DEFINE_integer("netvlad_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')

flags.DEFINE_integer("netvlad_dimred", -1,
                     "NetVLAD output dimension reduction")

flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                     "GatedNetVLAD output dimension reduction")

flags.DEFINE_bool("gating", False,
                  "Gating for NetVLAD")

flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")

flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")

flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_bool("fv_relu", True,
                  "ReLU after the NetFV hidden layer.")

flags.DEFINE_bool("fv_couple_weights", True,
                  "Coupling cluster weights or not")

flags.DEFINE_float("fv_coupling_factor", 0.01,
                   "Coupling factor")

class attention_frames_v0(models.BaseModel):
    def create_model(self,
                     model_input,
                     vocab_size,
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
        activation_video = slim.fully_connected(video_bn, 2048, activation_fn=None, biases_initializer=None)
        activation_audio = slim.fully_connected(audio_bn, 256, activation_fn=None, biases_initializer=None)

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

        # Compute Gates
        gate_video = slim.fully_connected(video_bn, 1, activation_fn=None, biases_initializer=None)
        gate_audio = slim.fully_connected(audio_bn, 1, activation_fn=None, biases_initializer=None)

        # Apply activations and reduce values
        video_vals = tf.transpose(activation_video, (2, 0, 1)) * tf.nn.softmax(gate_video[:, :, 0])
        video_vals = tf.reduce_sum(tf.transpose(video_vals, (1, 2, 0)), axis=1)

        audio_vals = tf.transpose(activation_audio, (2, 0, 1)) * tf.nn.softmax(gate_audio[:, :, 0])
        audio_vals = tf.reduce_sum(tf.transpose(audio_vals, (1, 2, 0)), axis=1)

        frame_agg = tf.concat([video_vals, audio_vals], axis=1)


        aggregated_model = getattr(video_level_models,
                                   "LogisticModel")
        return aggregated_model().create_model(
            model_input=frame_agg,
            vocab_size=vocab_size,
            **unused_params)


# Model source: https://github.com/antoine77340/Youtube-8M-WILLOW
class NetVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, self.feature_size, self.cluster_size],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(self.feature_size)))

        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad

class LightVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        vlad = tf.matmul(activation, reshaped_input)

        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad

class NetVLADModelLF(models.BaseModel):
    """Creates a NetVLAD based model.
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
        add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.netvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
        relu = FLAGS.netvlad_relu
        dimred = FLAGS.netvlad_dimred
        gating = FLAGS.gating
        remove_diag = FLAGS.gating_remove_diag
        lightvlad = FLAGS.lightvlad
        vlagd = FLAGS.vlagd

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

        if lightvlad:
            video_NetVLAD = LightVLAD(1024, max_frames, cluster_size, add_batch_norm, is_training)
            audio_NetVLAD = LightVLAD(128, max_frames, cluster_size / 2, add_batch_norm, is_training)
        elif vlagd:
            video_NetVLAD = NetVLAGD(1024, max_frames, cluster_size, add_batch_norm, is_training)
            audio_NetVLAD = NetVLAGD(128, max_frames, cluster_size / 2, add_batch_norm, is_training)
        else:
            video_NetVLAD = NetVLAD(1024, max_frames, cluster_size, add_batch_norm, is_training)
            audio_NetVLAD = NetVLAD(128, max_frames, cluster_size / 2, add_batch_norm, is_training)

        if add_batch_norm:  # and not lightvlad:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_VLAD"):
            vlad_video = video_NetVLAD.forward(reshaped_input[:, 0:1024])

        with tf.variable_scope("audio_VLAD"):
            vlad_audio = audio_NetVLAD.forward(reshaped_input[:, 1024:])

        vlad = tf.concat([vlad_video, vlad_audio], 1)

        vlad_dim = vlad.get_shape().as_list()[1]
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

        activation = tf.matmul(vlad, hidden1_weights)

        if add_batch_norm and relu:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn")

        else:
            hidden1_biases = tf.get_variable("hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases

        if relu:
            activation = tf.nn.relu6(activation)

        if gating:
            gating_weights = tf.get_variable("gating_weights_2",
                                             [hidden1_size, hidden1_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(hidden1_size)))

            gates = tf.matmul(activation, gating_weights)

            if remove_diag:
                # removes diagonals coefficients
                diagonals = tf.matrix_diag_part(gating_weights)
                gates = gates - tf.multiply(diagonals, activation)

            if add_batch_norm:
                gates = slim.batch_norm(
                    gates,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="gating_bn")
            else:
                gating_biases = tf.get_variable("gating_biases",
                                                [cluster_size],
                                                initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
                gates += gating_biases

            gates = tf.sigmoid(gates)

            activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)
