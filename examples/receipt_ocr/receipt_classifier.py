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

import glob
import os
import shutil

from vitaflow.helpers.print_helper import print_info


class ReceiptClassifier:
    def __init__(self,
                 text_files_in_dir,
                 out_dir):
        self._text_files_in_dir = text_files_in_dir
        self._out_dir = out_dir
        self._receipt_text_files_dir  = os.path.join(self._out_dir, "receipts")
        self._non_receipt_text_files_dir  = os.path.join(self._out_dir, "non_receipts")

        if not os.path.exists(self._receipt_text_files_dir):
            os.makedirs(self._receipt_text_files_dir)

        if not os.path.exists(self._non_receipt_text_files_dir):
            os.makedirs(self._non_receipt_text_files_dir)

    def naive_classifier(self, text):
        if "bill" in text or "receipt" in text or "$" in text:
            return True
        else:
            return False

    def classify(self, text_file_path):
        fd = open(text_file_path,"r")
        text = fd.read()
        return self.naive_classifier(text)

    def run(self):
        text_files_path = glob.glob(self._text_files_in_dir + os.sep+"*.txt")
        for text_file_path in text_files_path:
            print_info("Processing {}".format(text_file_path))
            is_receipt_file = self.classify(text_file_path=text_file_path)
            if is_receipt_file:
                shutil.copy(text_file_path, self._receipt_text_files_dir)
            else:
                shutil.copy(text_file_path, self._non_receipt_text_files_dir)

