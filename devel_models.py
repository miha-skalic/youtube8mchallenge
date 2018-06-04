"""
Models used for development.
"""

import math

import models
import video_level_models
import tensorflow as tf

import tensorflow.contrib.slim as slim


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
