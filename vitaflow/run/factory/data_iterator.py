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


class DataIteratorFactory():
    """
    Factory class to import the data iterators dynamically.
    This is used in conjuction with experiments, where dataset can be plugged in
    with the change of the filename
    """

    iterator_path = {
        "conll_csv_iterator": "vitaflow.data.text.iterators.conll_csv_iterator",
        "conll_csv_in_memory": "vitaflow.data.text.iterators.conll_csv_in_memory"
    }

    iterators = {
        "conll_csv_iterator": "CoNLLCsvDataIterator",
        "conll_csv_in_memory": "CoNLLCsvInMemory"

    }

    def __init__(self):
        pass

    @staticmethod
    def _get_iterator(name):
        """
        Finds the package and gets the class handle for the data iterator file name
        :param name:
        :return:
        """
        try:
            data_iterator = getattr(import_module(DataIteratorFactory.iterator_path[name]),
                                    DataIteratorFactory.iterators[name])
        except KeyError:
            raise NotImplemented("Given data iterator file name not found: {}".format(name))
        # Return the model class
        return data_iterator

    @staticmethod
    def get(iterator_name):
        """
        Returns the class handle for the given file name
        :param dataset_file_name:
        :return: class handle
        """
        iterator = DataIteratorFactory._get_iterator(iterator_name)
        return iterator

    @staticmethod
    def get_supported_data_iterators(dataset_type):
        possible_data_iterators = []
        for iterator_name in DataIteratorFactory.iterators:
            instance = DataIteratorFactory.get(iterator_name=iterator_name)
            if dataset_type in instance.dataset_types:
                possible_data_iterators.append(iterator_name)

        return possible_data_iterators
