
import importlib
import sys
import os
import tensorflow as tf

from vitaflow.run.experiments import Experiments

flags = tf.flags
flags.DEFINE_string("config_python_file", "config_python_file", "The config to use.")
flags.DEFINE_string("mode", "train", "train/retrain/predict")
FLAGS = flags.FLAGS
config = FLAGS.config_python_file.replace("/", ".")
config = config.replace(".py", "")
config = importlib.import_module(config)

if __name__ == "__main__":
    experiment = Experiments(hparams=config.experiments, mode=FLAGS.mode)
    experiment.run()