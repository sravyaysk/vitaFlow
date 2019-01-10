# Copyright 2019 The Shabda Authors. All Rights Reserved.
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
REsseract based OCR
"""

import pytesseract
import cv2
import re
import os
import glob
import concurrent.futures
import time
from tqdm import tqdm
from wand.image import Image
from PIL import Image as PI
import pyocr
import pyocr.builders
import io

from vitaflow.helpers.print_helper import print_info


class TesseractOCR:
    def __init__(self,
                 image_dir,
                 text_out_dir):
        self._image_dir = image_dir
        self._text_out_dir = text_out_dir

    def convert_image(self, image_path):
        img = cv2.imread(image_path)
        text = pytesseract.image_to_string(img,lang='eng',config='--psm 6')
        image_path = os.path.normpath(image_path)
        file_name = image_path.split(os.sep)[-1].split(".")[0]
        text_file_path = os.path.join(self._text_out_dir, file_name+".txt")
        fd = open(text_file_path,"w")
        fd.write("%s" % text)
        return text_file_path

    def convert_pdf(self, pdf_path):
        """
        Reference: https://pythontips.com/2016/02/25/ocr-on-pdf-files-using-python/
        :param pdf_path:
        :return:
        """

        tool = pyocr.get_available_tools()[0]
        lang = tool.get_available_languages()[0]

        print_info(tool.get_available_languages())
        pdf_path = os.path.normpath(pdf_path)
        file_name = pdf_path.split(os.sep)[-1].split(".")[0]

        # with Image(filename=pdf_path, resolution=300) as img:
        #     img.compression_quality = 99
        #     img.save(filename=os.path.join(self._image_dir,file_name))

        req_image = []
        final_text = []
        text_file_path = ""

        image_pdf = Image(filename=pdf_path, resolution=300)
        image_jpeg = image_pdf.convert('jpeg')
        for img in image_jpeg.sequence:
            img_page = Image(image=img)
            req_image.append(img_page.make_blob('jpeg'))

        for i, img in tqdm(enumerate(req_image)):
            text = tool.image_to_string(
                PI.open(io.BytesIO(img)),
                lang=lang,
                builder=pyocr.builders.TextBuilder()
            )
            text_file_path = os.path.join(self._text_out_dir, file_name+str(i)+".txt")
            fd = open(text_file_path,"w")
            fd.write("%s" % text)
        return text_file_path

    def convert(self, path):
        print_info(path)
        if path.endswith("pdf"):
            return self.convert_pdf(pdf_path=path)
        else:
            return self.convert_image(image_path=path)


    def parallel_convert(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            image_list = glob.glob(self._image_dir+os.sep + "*.png")
            image_list.extend(glob.glob(self._image_dir+os.sep + "*.pdf"))
            print_info(image_list)
            for img_path,out_file in zip(image_list, executor.map(self.convert, image_list)):
                print(img_path,',',out_file,', processed')




