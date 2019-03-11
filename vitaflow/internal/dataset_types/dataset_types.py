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
Dataset type classes
"""


class IDatasetType():
    """
    `Datasets that follow CoNLL format <http://universaldependencies.org/docs/format.html>`_.

    """
    
    def __init__(self):
        pass

    @property
    def dataset_type(self):
        """
        Inheriting class must provide its name as string
        :return: Name of the class as string
        """
        return type(self).__name__


class ICSVSeq2SeqType1(IDatasetType):
    """
    Datasets with `WORD ENTITY_COL1 ENTITY_COL2 ENTITY_COL3`.
    There will be three labels for given sentence which is broken into words.
    .. code-block:: text

        By IN B-PP O
        stumps NNS B-NP O
        Kent NNP B-NP B-ORG
        had VBD B-VP O
        reached VBN I-VP O
        108 CD B-NP O
        for IN B-PP O
        three CD B-NP O
        . . O O

    Dataset consists of three CSV files for train//val/test respectively.
    Each CSV has four columns namely `WORD ENTITY_COL1 ENTITY_COL2 ENTITY_COL3`.
    """
    
    def __init__(self):
        IDatasetType.__init__(self)
        self.TEXT_COL = "0"
        self.ENTITY_COL1 = "1"
        self.ENTITY_COL2 = "2"
        self.ENTITY_COL3 = "3"

    @property
    def text_col(self):
        return self.TEXT_COL

    @text_col.setter
    def text_col(self, text_col):
        self.TEXT_COL = text_col

    @property
    def entity_col1(self):
        return self.ENTITY_COL1

    @entity_col1.setter
    def entity_col1(self, entity_col1):
        self.ENTITY_COL1 = entity_col1

    @property
    def entity_col2(self):
        return self.ENTITY_COL2

    @entity_col2.setter
    def entity_col2(self, entity_col2):
        self.ENTITY_COL2 = entity_col2

    @property
    def entity_col3(self):
        return self.ENTITY_COL3

    @entity_col3.setter
    def entity_col3(self, entity_col3):
        self.ENTITY_COL3 = entity_col3

#
# class ICLIENTXType1(IDatasetType):
#     """
#     Datasets with `WORD ENTITY_COL1 ENTITY_COL2 ENTITY_COL3`.
#     There will be three labels for given sentence which is broken into words.
#     .. code-block:: text
#
#         By IN B-PP O
#         stumps NNS B-NP O
#         Kent NNP B-NP B-ORG
#         had VBD B-VP O
#         reached VBN I-VP O
#         108 CD B-NP O
#         for IN B-PP O
#         three CD B-NP O
#         . . O O
#
#     Dataset consists of three CSV files for train//val/test respectively.
#     Each CSV has four columns namely `WORD ENTITY_COL1 ENTITY_COL2 ENTITY_COL3`.
#     """
#
#     def __init__(self):
#         IDatasetType.__init__(self)
#         self.TEXT_COL = "word"
#         self.ENTITY_COL1 = "1"
#         self.ENTITY_COL2 = "2"
#         self.ENTITY_COL3 = "3"
#
#     @property
#     def text_col(self):
#         return self.TEXT_COL
#
#     @text_col.setter
#     def text_col(self, text_col):
#         self.TEXT_COL = text_col
#
#     @property
#     def entity_col1(self):
#         return self.ENTITY_COL1
#
#     @entity_col1.setter
#     def entity_col1(self, entity_col1):
#         self.ENTITY_COL1 = entity_col1
#
#     @property
#     def entity_col2(self):
#         return self.ENTITY_COL2
#
#     @entity_col2.setter
#     def entity_col2(self, entity_col2):
#         self.ENTITY_COL2 = entity_col2
#
#     @property
#     def entity_col3(self):
#         return self.ENTITY_COL3
#
#     @entity_col3.setter
#     def entity_col3(self, entity_col3):
#         self.ENTITY_COL3 = entity_col3
