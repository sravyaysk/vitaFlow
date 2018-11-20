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
Factory class to get data iterator for experiments
"""

import sys
from importlib import import_module



class ModelsFactory():
    """
    Factory class to import the models dynamically.
    This is used in conjuction with experiments, where dataset can be plugged in
    with the change of the filename
    """
    model_path = {
        "bilstm_crf" : "vitaflow.models.text.seq2seq.bilstm_crf",
    }

    models = {
        "bilstm_crf" : "BiLSTMCrf"
    }


    def __init__(self):
        pass

    @staticmethod
    def _get_model(name):
        """
        Finds the package and gets the class handle for the data iterator file name
        :param name:
        :return:
        """
        try:
            model = getattr(import_module(ModelsFactory.model_path[name]), ModelsFactory.models[name])
        except KeyError:
            raise NotImplemented("Given config file name not found: {}".format(name))
        # Return the model class
        return model


    @staticmethod
    def get(model_name):
        """
        Returns the class handle for the given file name
        :param dataset_file_name:
        :return: class handle
        """
        model = ModelsFactory._get_model(model_name)
        return model

    @staticmethod
    def get_supported_models(data_iterator_type):
        possible_model_names = []
        for model_name in ModelsFactory.models:
            instance = ModelsFactory.get(model_name=model_name)
            if data_iterator_type in instance.data_iterator_types:
                possible_model_names.append(model_name)

        return possible_model_names