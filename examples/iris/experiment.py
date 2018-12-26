#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Executor script for Con112003 Dataset


Command:
    python examples/iris/run.py --config_python_file=examples/iris/config.py
"""
import importlib
import os
import sys

import tensorflow as tf

# Appending vitaFlow main Path
sys.path.append(os.path.abspath('.'))

from vitaflow.run.experiments import Experiments
from pprint import pprint

# TODO: Use ArgParse
# tf.args works just like argparse
flags = tf.flags
flags.DEFINE_string("config_python_file", "config_python_file", "The config to use.")
flags.DEFINE_string("mode", "train", "train/retrain/predict")
FLAGS = flags.FLAGS
config = FLAGS.config_python_file.replace("/", ".")
config = config.replace(".py", "")
config = importlib.import_module(config)

pprint(config)


# TODO: add iris_iterator

if __name__ == "__main__":
    print(' -' * 35)
    print('Running Experiment: Conll')
    print(' -' * 35)
    import pdb
    pdb.set_trace()
    experiment = Experiments(hparams=config.experiments, mode=FLAGS.mode)
    experiment.run()
    print(' -' * 35)
