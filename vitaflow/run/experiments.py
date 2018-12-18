# Copyright 2018 The Shabda Authors. All Rights Reserved.
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
Experiments class that allows easy plug n play of modules
"""
import logging
from importlib import import_module

import tensorflow as tf
from tqdm import tqdm

from vitaflow.core.hyperparams import HParams
from vitaflow.helpers.print_helper import *
from vitaflow.run.executor import Executor

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

CGREEN2 = '\33[92m'
CEND = '\33[0m'


class Experiments(object):
    """
    Experiments uses dataset, data iterator & model factory classes and import them
    dynamically based on the string.
    This allows the user to choose the modules dynamically and run the experiments without ever writing the
    code when we need mix and experiment dataset and modules.
    """

    def __init__(self, hparams, mode='train'):
        self._hparams = HParams(hparams, self.default_hparams(), allow_new_hparam=True)

        self.mode = mode

        self._dataset = None
        self.data_iterator = None
        self._model = None

    @staticmethod
    def default_hparams():
        return {
            "num_epochs": 5,
            "dataset_class_with_path": None,
            "iterator_class_with_path": None,
            "model_class_with_path": None,
            "save_checkpoints_steps" : 50,
            "keep_checkpoint_max" : 5,
            "save_summary_steps" : 25,
            "log_step_count_steps" : 10
        }

    def _get_class(self, package, name):
        """
        Import the givenpackage and the class dynmaically
        :param package: Pacakage path of the class
        :param name: Name of the class
        :return: Instance of the class object
        """
        return getattr(import_module(package), name)

    def get_dataset_reference(self, dataset_class_with_path):
        """
        Uses the dataset name to get the reference from the dataset factory class
        :param dataset_class_with_path:
        :return:
        Eg: vitaflow.data.text.conll.conll_2003_dataset.CoNLL2003Dataset
        """

        print_debug("Dynamically importing dataset : " + dataset_class_with_path)
        package, name = dataset_class_with_path.rsplit(".", 1)
        # dataset = DatasetFactory.get(dataset_file_name=dataset_name)
        dataset = self._get_class(package=package, name=name)
        return dataset

    def get_iterator_reference(self, iterator_class_with_path):
        """
        Uses the iterator name to get the reference from the iterator factory class
        :param iterator_class_with_path:
        :return:
        """

        print_debug("Dynamically importing iterator : " + iterator_class_with_path)
        # iterator = DataIteratorFactory.get(iterator_name=iterator_name)
        package, name = iterator_class_with_path.rsplit(".", 1)
        iterator = self._get_class(package=package, name=name)
        return iterator

    def get_model_reference(self, model_class_with_path):
        """
        Uses the model name to get the reference from the model factory class
        :param model_class_with_path:
        :return:
        """

        print_debug("Dynamically importing model : " + model_class_with_path)
        package, name = model_class_with_path.rsplit(".", 1)
        model = self._get_class(package=package, name=name)
        # model = ModelsFactory.get(model_name=model_name)
        return model

    def check_interoperability_n_import(self):
        #Using factory classes get the handle for the actual classes from string
        self._dataset = self.get_dataset_reference(self._hparams['dataset_class_with_path'])
        self._data_iterator = self.get_iterator_reference(self._hparams['iterator_class_with_path'])
        self._model = self.get_model_reference(self._hparams['model_class_with_path'])

        # if not self._data_iterator.dataset_type == self._dataset.dataset_type:
        #     print_info("Possible data iterators are: {}".
        #                format(DataIteratorFactory.get_supported_data_iterators(self._dataset.dataset_type)))
        #     raise RuntimeError("Selected data iterator and data set can't be used together")
        #
        # if not self._model.data_iterator_type == self._data_iterator.data_iterator_type:
        #     print_info("Possible models are: {}".
        #                format(ModelsFactory.get_supported_data_iterators(self._dataset.dataset_type)))
        #     raise RuntimeError("Selected model and data iterator can't be used together")

    def _init_tf_config(self):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        # run_config.gpu_options.per_process_gpu_memory_fraction = 0.50
        run_config.allow_soft_placement = True
        run_config.log_device_placement = False
        model_dir = self._model.model_dir
        self._run_config = tf.estimator.RunConfig(session_config=run_config,
                                                  save_checkpoints_steps=self._hparams.save_checkpoints_steps,
                                                  keep_checkpoint_max=self._hparams.keep_checkpoint_max,
                                                  save_summary_steps=self._hparams.save_summary_steps,
                                                  model_dir=model_dir,
                                                  log_step_count_steps=self._hparams.log_step_count_steps)
        return run_config

    def setup(self):
        self.check_interoperability_n_import()
        # Initialize the handles and call any user specific init() methods
        self._dataset = self._dataset(hparams=self._hparams[self._hparams['dataset_class_with_path']])
        self._data_iterator = self._data_iterator(hparams=self._hparams[self._hparams['iterator_class_with_path']],
                                                  dataset=self._dataset)
        self._model = self._model(hparams=self._hparams[self._hparams['model_class_with_path']], data_iterator=self._data_iterator)

    def test_dataset(self):
        iterator = self._data_iterator.test_sentence_input_fn("@ º &").make_initializable_iterator()
        next_element = iterator.get_next()
        # print_debug(next_element)
        init_op = iterator.initializer
        with tf.Session() as sess:
            # Initialize the iterator
            sess.run(init_op)
            # print(sess.run(next_element))
            res = sess.run(next_element)
            print_debug(res[0])
            print_debug(res[1])
            # Move the iterator back to the beginning
            # sess.run(init_op)
            # print(sess.run(next_element))

    def run(self):
        self.setup()

        num_samples = self._data_iterator.num_train_samples
        batch_size = self._hparams[self._hparams['iterator_class_with_path']].batch_size
        num_epochs = self._hparams.num_epochs
        mode = self.mode

        self._init_tf_config()

        exec = Executor(model=self._model, data_iterator=self._data_iterator, config=self._run_config)

        if (mode == "train" or mode == "retrain"):
            for current_epoch in tqdm(range(num_epochs), desc="Epoch"):
                current_max_steps = (num_samples // batch_size) * (current_epoch + 1)
                exec.train(max_steps=current_max_steps)  # , eval_steps=None)
                exec.evaluate(steps=200)

        elif (mode == "predict"):
            exec.predict()

        elif (mode == "predict_sentence"):
            exec.predict_sentence("SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .")
