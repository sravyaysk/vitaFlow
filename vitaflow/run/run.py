#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import os
import sys

import tensorflow as tf

# Appending vitaFlow main Path
sys.path.append(os.path.abspath('.'))

from vitaflow.run.experiments import Experiments

# TODO: Use ArgParse
# tf.args works just like argparse
flags = tf.flags
flags.DEFINE_string("config_python_file", "config_python_file", "The config to use.")
flags.DEFINE_string("mode", "train", "train/retrain/predict/predict_instance")
flags.DEFINE_string("test_file_path", "", "")

FLAGS = flags.FLAGS
config = FLAGS.config_python_file.replace("/", ".")
config = config.replace(".py", "")
config = importlib.import_module(config)


if __name__ == "__main__":
    print(' -' * 35)
    print('Running Experiment:',config.experiment_name)
    print(' -' * 35)
    import spacy

    try:
        spacy.load('en_core_web_md')
    except OSError:
        download_en_core_web_md = 'python -m spacy download en_core_web_md'
        import subprocess

        subprocess.call(download_en_core_web_md.split())
        print('OSError: Not able to find Spacy Package - Downloading "en_core_web_md" !')
        print('OSError: Run following command & re-run the experiment')
        print(' -' * 35)
        print('Downloading missing package(command: {})'.format(download_en_core_web_md))
        print(' -' * 35)
    experiment = Experiments(hparams=config.experiments, mode=FLAGS.mode)
    experiment.run(args=FLAGS)
    print(' -' * 35)
