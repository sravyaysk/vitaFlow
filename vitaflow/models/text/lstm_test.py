# coding=utf-8
# Copyright 2019 The vitaFlow Authors. All Rights Reserved.
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
# Forked from https://github.com/tensorflow/tensor2tensor

"""LSTMSeq2Seq models tests."""

import numpy as np

from vitaflow.internal.data_generators import problem_hparams
from vitaflow.models.text import lstm

import tensorflow as tf

from vitaflow.utils.print_helper import print_error


class LSTMTest(tf.test.TestCase):

  def testLSTMSeq2Seq(self):
    vocab_size = 9
    x = np.random.randint(1, high=vocab_size, size=(3, 5, 1, 1))
    y = np.random.randint(1, high=vocab_size, size=(3, 6, 1, 1))
    hparams = lstm.lstm_seq2seq()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size,
                                                     vocab_size,
                                                     hparams)
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = lstm.LSTMSeq2seq(hparams, tf.estimator.ModeKeys.TRAIN,
                               p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (3, 6, 1, 1, vocab_size))

  def testLSTMSeq2SeqAttention(self):
    vocab_size = 9
    x = np.random.randint(1, high=vocab_size, size=(3, 5, 1, 1))
    y = np.random.randint(1, high=vocab_size, size=(3, 6, 1, 1))
    hparams = lstm.lstm_attention()

    p_hparams = problem_hparams.test_problem_hparams(vocab_size,
                                                     vocab_size,
                                                     hparams)
    x = tf.constant(x, dtype=tf.int32)
    x = tf.placeholder_with_default(x, shape=[None, None, 1, 1])

    with self.test_session() as session:
      features = {
          "inputs": x,
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = lstm.LSTMSeq2seqAttention(
          hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (3, 6, 1, 1, vocab_size))

  def testLSTMSeq2seqBidirectionalEncoder(self):
    vocab_size = 9
    x = np.random.randint(1, high=vocab_size, size=(3, 5, 1, 1))
    y = np.random.randint(1, high=vocab_size, size=(3, 6, 1, 1))
    hparams = lstm.lstm_seq2seq()
    p_hparams = problem_hparams.test_problem_hparams(vocab_size,
                                                     vocab_size,
                                                     hparams)
    with self.test_session() as session:
      features = {
          "inputs": tf.constant(x, dtype=tf.int32),
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = lstm.LSTMSeq2seqBidirectionalEncoder(
          hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (3, 6, 1, 1, vocab_size))

  def testLSTMSeq2seqAttentionBidirectionalEncoder(self):
    vocab_size = 9
    x = np.random.randint(1, high=vocab_size, size=(3, 5, 1, 1))
    y = np.random.randint(1, high=vocab_size, size=(3, 6, 1, 1))
    hparams = lstm.lstm_attention()

    p_hparams = problem_hparams.test_problem_hparams(vocab_size, vocab_size, hparams) #TODO note: hparams added here
    x = tf.constant(x, dtype=tf.int32)
    x = tf.placeholder_with_default(x, shape=[None, None, 1, 1])

    with self.test_session() as session:
      features = {
          "inputs": x,
          "targets": tf.constant(y, dtype=tf.int32),
      }
      model = lstm.LSTMSeq2seqAttentionBidirectionalEncoder(
          hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
      logits, _ = model(features)
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (3, 6, 1, 1, vocab_size))


if __name__ == "__main__":
  tf.test.main()
