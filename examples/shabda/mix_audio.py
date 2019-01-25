#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import os
import sys
import librosa
import numpy as np
import tensorflow as tf

# Appending vitaFlow main Path
sys.path.append(os.path.abspath('.'))

from vitaflow.run.experiments import Experiments

# TODO: Use ArgParse
# tf.args works just like argparse
user_home = os.path.join(os.path.expanduser("~"), "vitaFlow/")
flags = tf.flags
flags.DEFINE_string("file1", user_home + "/TPS/preprocessed_data/test/MD/MD24_04.wav", "Path to audio file 1")
flags.DEFINE_string("file2",user_home + "/TPS/preprocessed_data/test/ML/ML67_01.wav", "Path to audio file 2")
flags.DEFINE_string("outfile", "mix.wav", "Path to output file")
flags.DEFINE_integer("sampling_rate", 8000, "Sampling Rate")
FLAGS = flags.FLAGS



if __name__ == "__main__":
    print(' -' * 35)
    print('Mixing audio')
    print(' -' * 35)


    data1, _ = librosa.load(FLAGS.file1, sr=FLAGS.sampling_rate)
    data2, _ = librosa.load(FLAGS.file2, sr=FLAGS.sampling_rate)

    mix = data1[:len(data2)] + data2[:len(data1)]
    librosa.output.write_wav(FLAGS.outfile, mix, FLAGS.sampling_rate)

