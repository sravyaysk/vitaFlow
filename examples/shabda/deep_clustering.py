# Copyright 2018 The vitaFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Class Model: model for the deep clustering speech seperation
"""

import os
from overrides import overrides
import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib import lookup

from examples.shabda.core.feature_types.shabda_wav_pair_feature import ShabdaWavPairFeature
from vitaflow.core.hyperparams import HParams
from vitaflow.core.models.model_base import ModelBase
from vitaflow.helpers.print_helper import print_error


class DeepClustering(ModelBase, ShabdaWavPairFeature):
    def __init__(self, hparams=None, data_iterator=None):
        # ITextFeature.__init__(self)
        ModelBase.__init__(self, hparams=hparams)
        ShabdaWavPairFeature.__init__(self)
        self._hparams = HParams(hparams,
                            self.default_hparams())

        print_error(self._hparams)
        self.n_hidden = self._hparams.n_hidden
        self.batch_size = self._hparams.batch_size
        self.p_keep_ff = self._hparams.p_keep_ff
        self.p_keep_rc = self._hparams.p_keep_rc
        self.neff = self._hparams.neff
        self.embd_dim = self._hparams.embd_dim
        self.frames_per_sample = self._hparams.frames_per_sample

        self.weights = None
        self.biases = None

    @staticmethod
    def default_hparams():
        params = ModelBase.default_hparams()
        params.update({
            "n_hidden" : 4,
            "batch_size" : 32,
            "p_keep_ff" : 0.5,
            "p_keep_rc" : 0.5,
            "frames_per_sample" : 100,
            "embd_dim" : 30,
            "neff" : 129
        })
        return params

    @overrides
    def _get_loss(self, labels, logits, VAD):
        labels = tf.cast(labels, tf.float32)
        VAD = tf.cast(VAD, tf.float32) #[batch_size, FPS, NEFF]
        Y = labels  # [batch_size, FPS, NEFF, 2]
        embeddings = logits  # [batch_size * FPS, NEFF, embd_dim]

        Y = tf.reshape(Y, [self.batch_size*self.frames_per_sample, self.neff, 2]) # [batch_size * FPS, NEFF, 2]
        # [batch_size * FPS, NEFF]
        VAD_reshaped = tf.reshape(VAD, [self.batch_size*self.frames_per_sample, self.neff], name="VAD_reshaped")

        '''Defining the loss function'''
        embeddings_rs = tf.reshape(embeddings, shape=[self.batch_size*self.frames_per_sample*self.neff, self._hparams.embd_dim]) # [batch_size * FPS * NEFF, embd_dim]

        # [batch_size * FPS * NEFF]
        VAD_rs = tf.reshape(VAD_reshaped, shape=[self.batch_size*self.frames_per_sample*self.neff], name="VAD_rs")

        # get the embeddings with active VAD
        # [embd_dim, batch_size * FPS * NEFF]  x  [batch_size * FPS * NEFF]
        #  [batch_size * FPS * NEFF, embd_dim]
        embeddings_rsv = tf.transpose(tf.multiply(tf.transpose(embeddings_rs), VAD_rs))
        #[batch_size,  FPS * NEFF, embd_dim]
        embeddings_v = tf.reshape(embeddings_rsv, [self.batch_size,
                                                   self._hparams.frames_per_sample * self._hparams.neff,
                                                   self._hparams.embd_dim])

        # get the Y(speaker indicator function) with active VAD
        Y_rs = tf.reshape(Y, shape=[self.batch_size*self.frames_per_sample*self.neff, 2], name="Y_rs") # [batch_size * FPS * NEFF, 2]
        # [batch_size * FPS * NEFF, 2] x [batch_size * FPS * NEFF]
        Y_rsv = tf.transpose(tf.multiply(tf.transpose(Y_rs), VAD_rs))
        # [batch_size, FPS * NEFF, 2]
        Y_v = tf.reshape(Y_rsv, shape=[self.batch_size, self._hparams.frames_per_sample * self._hparams.neff, 2], name="Y_v")

        # fast computation format of the embedding loss function
        embeddings_v_t = tf.transpose(embeddings_v, [0, 2, 1]) # [batch_size, embd_dim, FPS * NEFF]
        Y_v_t = tf.transpose(Y_v, [0, 2, 1]) #[batch_size, 2, FPS * NEFF]

        # [batch_size,  embd_dim, FPS * NEFF] x [batch_size,  FPS * NEFF, embd_dim] -
        loss_1 = tf.matmul(embeddings_v_t, embeddings_v) # [batch_size, embd_dim, embd_dim]

        # [batch_size,  FPS * NEFF, embd_dim] x [batch_size, FPS * NEFF, 2] +
        loss_2 = tf.matmul(embeddings_v_t, Y_v) # [batch_size, embd_dim, 2]

        # [batch_size, 2, FPS * NEFF] x [batch_size, FPS * NEFF, 2]
        loss_3 = tf.matmul(Y_v_t, Y_v) # [batch_size, 2, 2]

        loss_batch = tf.nn.l2_loss(loss_1) - 2 * tf.nn.l2_loss(loss_2) + tf.nn.l2_loss(loss_3)

        loss_v = (loss_batch) / self.batch_size / (self._hparams.frames_per_sample^2)
        tf.summary.scalar('loss', loss_v)

        tf.logging.info("embeddings: =====> {}".format(embeddings))
        tf.logging.info("Y: =====> {}".format(Y))
        tf.logging.info("VAD: =====> {}".format(VAD))
        tf.logging.info("VAD_reshaped: =====> {}".format(VAD_reshaped))


        tf.logging.info("embeddings_rs: =====> {}".format(embeddings_rs))
        tf.logging.info("VAD_rs: =====> {}".format(VAD_rs))
        tf.logging.info("embeddings_rsv: =====> {}".format(embeddings_rsv))
        tf.logging.info("embeddings_v: =====> {}".format(embeddings_v))

        tf.logging.info("Y_rs: =====> {}".format(Y_rs))
        tf.logging.info("Y_rsv: =====> {}".format(Y_rsv))
        tf.logging.info("Y_v: =====> {}".format(Y_v))

        tf.logging.info("embeddings_v_t: =====> {}".format(embeddings_v_t))
        tf.logging.info("Y_v_t: =====> {}".format(Y_v_t))
        tf.logging.info("loss_1: =====> {}".format(loss_1))
        tf.logging.info("loss_2: =====> {}".format(loss_2))
        tf.logging.info("loss_3: =====> {}".format(loss_3))

        tf.logging.info("loss_batch: =====> {}".format(loss_batch))


        tf.logging.info("loss_v: =====> {}".format(loss_v))


        return loss_v

    def _build_layers(self, features, mode):
        '''The structure of the network'''
        # four layer of LSTM cell blocks

        tf.logging.info(" =====> features {}".format(features))

        tf.logging.info(" =====> BLSTM1")
        with tf.variable_scope('BLSTM1') as scope:
            lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden,
                layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell,
                input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden,
                layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell,
                input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell,
                lstm_bw_cell,
                features,
                sequence_length=[self._hparams.frames_per_sample] * self.batch_size,
                dtype=tf.float32)
            state_concate = tf.concat(outputs, 2)
            tf.logging.info(" =====> outputs {}".format(outputs))

        tf.logging.info(" =====> BLSTM2")
        with tf.variable_scope('BLSTM2') as scope:
            # lstm_fw_cell2 = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            # lstm_bw_cell2 = tf.nn.rnn_cell.LSTMCell(
            #     self.n_hidden)
            lstm_fw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden,
                layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell2,
                input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell2 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden,
                layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell2 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell2,
                input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs2, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell2,
                lstm_bw_cell2,
                state_concate,
                sequence_length=[self._hparams.frames_per_sample] * self.batch_size,
                dtype=tf.float32)
            tf.logging.info(" =====> outputs2 {}".format(outputs2))

            state_concate2 = tf.concat(outputs2, 2)

        tf.logging.info(" =====> BLSTM3")
        with tf.variable_scope('BLSTM3') as scope:
            lstm_fw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden,
                layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell3 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell3,
                input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell3 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden,
                layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell3 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell3,
                input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs3, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell3,
                lstm_bw_cell3,
                state_concate2,
                sequence_length=[self._hparams.frames_per_sample] * self.batch_size,
                dtype=tf.float32)

            tf.logging.info(" =====> outputs3 {}".format(outputs3))

            state_concate3 = tf.concat(outputs3, 2)

        tf.logging.info(" =====> BLSTM4")
        with tf.variable_scope('BLSTM4') as scope:
            lstm_fw_cell4 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden,
                layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_fw_cell4 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_fw_cell4,
                input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            lstm_bw_cell4 = tf.contrib.rnn.LayerNormBasicLSTMCell(
                self.n_hidden, layer_norm=False,
                dropout_keep_prob=self.p_keep_rc)
            lstm_bw_cell4 = tf.nn.rnn_cell.DropoutWrapper(
                lstm_bw_cell4,
                input_keep_prob=1,
                output_keep_prob=self.p_keep_ff)
            outputs4, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_cell4, lstm_bw_cell4, state_concate3,
                sequence_length=[self._hparams.frames_per_sample] * self.batch_size,
                dtype=tf.float32)

            tf.logging.info(" =====> outputs4{}".format(outputs4))

            state_concate4 = tf.concat(outputs4, 2)


            # one layer of embedding output with tanh activation function
        out_concate = tf.reshape(state_concate4, [-1, self.n_hidden * 2])

        tf.logging.info("state_concate4: =====> {}".format(state_concate4))
        tf.logging.info("out_concate: =====> {}".format(out_concate))

        emb_out = tf.matmul(out_concate,
                            self.weights['out']) + self.biases['out']
        emb_out = tf.nn.tanh(emb_out)

        tf.logging.info("emb_out: =====> {}".format(emb_out))

        reshaped_emb = tf.reshape(emb_out, [-1, self._hparams.neff, self._hparams.embd_dim])
        # normalization before output
        normalized_emb = tf.nn.l2_normalize(reshaped_emb, 2)

        tf.logging.info("normalized_emb: =====> {}".format(normalized_emb))

        return normalized_emb

    def _get_predicted_classes(self, logits):
        raise NotImplementedError

    def _get_class_probabilities(self, logits):
        raise NotImplementedError

    def _get_optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 200)
        train_op = optimizer.apply_gradients(
            zip(gradients, v),
            global_step=tf.train.get_global_step())

        return train_op

    def _get_eval_metrics(self, predictions, labels):
        raise NotImplementedError

    def _build(self, features, labels, params, mode, config=None):
        """
        Used for the :tf_main:`model_fn <estimator/Estimator#__init__>`
        argument when constructing
        :tf_main:`tf.estimator.Estimator <estimator/Estimator>`.
        """

        self.weights = {
            'out': tf.Variable(
                tf.random_normal([2 * self.n_hidden, self.embd_dim * self.neff]))
        }

        self.biases = {
            'out': tf.Variable(
                tf.random_normal([self.embd_dim * self.neff]))
        }

        samples = features[self.FEATURE_1_NAME]
        vad = features[self.FEATURE_2_NAME]
        tf.logging.info("samples: =====> {}".format(samples))
        tf.logging.info("vad: =====> {}".format(vad))


        embeddings = self._build_layers(samples, mode)

        loss = None
        optimizer = None
        if mode != tf.estimator.ModeKeys.PREDICT:
                # labels = tf.reshape(labels, shape=(-1, self._out_dim), name="labels")
            tf.logging.info('labels: -----> {}'.format(labels))

            loss = self._get_loss(labels=labels, logits=embeddings, VAD=vad)
            optimizer = self._get_optimizer(loss=loss)
            # eval_metric_ops = self._get_eval_metrics(predictions=predictions, labels=ner_ids)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=None,
            loss=loss,
            train_op=optimizer,
            eval_metric_ops=None)
