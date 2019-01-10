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
Main file thats puts different pieces together for the issue #14
"""
import os

from examples.receipt_ocr.receipt_classifier import ReceiptClassifier
from vitaflow.ocr.tesseract import TesseractOCR

experiment_root_directory = os.path.join(os.path.expanduser("~"), "vitaFlow")
image_dir = os.path.join(experiment_root_directory, "receipt", "dataset", "png")
text_out_dir = os.path.join(experiment_root_directory, "receipt", "dataset", "text")
classified_text_dir = os.path.join(experiment_root_directory, "receipt", "dataset", "classified")

ocr = TesseractOCR(image_dir=image_dir,
                   text_out_dir=text_out_dir)
classifier = ReceiptClassifier(text_files_in_dir=text_out_dir,
                               out_dir=classified_text_dir)
ocr.parallel_convert()
classifier.run()










