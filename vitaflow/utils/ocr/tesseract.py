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

import concurrent.futures
import glob
import io
import os

import cv2
import pyocr
import pyocr.builders
import pytesseract
from PIL import Image as PI
from tqdm import tqdm
from wand.image import Image

from vitaflow.utils.print_helper import print_info, print_error


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
        file_name = image_path.split(os.sep)[-2]
        tag_name = image_path.split(os.sep)[-1].split(".")[0]
        file_path = os.path.join(self._text_out_dir, file_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        text_file_path = os.path.join(file_path, tag_name+".txt")
        print_error(text_file_path)
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
            with open(text_file_path, "w") as fd:
                fd.write("%s" % text)
        return text_file_path

    def convert(self, path):
        print_info(path)
        if path.endswith("pdf"):
            return self.convert_pdf(pdf_path=path)
        else:
            return self.convert_image(image_path=path)


    def parallel_convert(self):
        print_info("Running OCR : {}".format(self._image_dir))
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            image_list = glob.glob(self._image_dir+ os.sep + "*/*.jpg")
            image_list.extend(glob.glob(self._image_dir+ os.sep + "*/*.jpeg"))
            image_list.extend(glob.glob(self._image_dir+ os.sep + "*/*.png"))

            # print_info(image_list)
            try:
                for img_path, out_file in zip(image_list, executor.map(self.convert, image_list)):
                    print(img_path, ',', out_file, ', processed')
            except:
                pass
