# coding=utf-8
from __future__ import unicode_literals

"""

to run
    `PYTHONIOENCODING=utf-8 python3`

"""
import os
import numpy as np
import matplotlib.pyplot as plt

import cv2
import config
from pytesseract import image_to_string


# Tesseract - Unicode errors
# encoding=utf8


def show_img(data):
    plt.figure(figsize=(10, 10))
    plt.imshow(data, 'gray')


def get_threshold_image(image):
    image = (image * 1.0) / image.max()
    THRESHOLD_LIMIT = 0.40
    image[image <= THRESHOLD_LIMIT] = 0.0
    image[image > THRESHOLD_LIMIT] = 1.0
    return image


def get_threshold_image2(image):
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)
    image = (image * 1.0) / image.max()
    THRESHOLD_LIMIT = 0.40
    image[image <= THRESHOLD_LIMIT] = 0.0
    image[image > THRESHOLD_LIMIT] = 1.0
    return image


def get_line_segments(image):
    # threshold
    image = get_threshold_image(image)
    plt_data = image.min(axis=1)
    # plt.figure(figsize=(15, 2))
    # plt.plot(range(len(plt_data)), plt_data, '*')

    plt_data_index = np.arange(len(plt_data))
    data = plt_data_index[plt_data == 0]
    i = 0
    start = i
    memory = data[i]

    line_segments = []

    while i < len(data) - 1:
        i += 1
        if data[i] == memory + 1:
            memory += 1
        else:
            line_segments.append(
                (data[start], data[i])
            )
            # print(data[start], data[i])
            start = i
            memory = data[i]

    return line_segments


def image_to_text_lines(image):
    line_segments = get_line_segments(image)
    collected_text_data = []
    for start, end in line_segments:
        if abs(start - end) < 10:
            continue
        text_data = image_to_string(image[start - 2: end + 2, :])
        if text_data:
            # show_img(image[start - 1: end + 1, :])
            # print(text_data)
            collected_text_data.append(text_data)
    return collected_text_data


def main(image_filename):
    '''
    convert image -> text file

    :param image_filename: filename with path
    :return:
    '''
    image = plt.imread(image_filename)
    _text_file_name = (os.path.basename(image_filename)).rsplit('.')[0] + '.txt'
    text_file_name = os.path.join(config.TEXT_DIR, _text_file_name)
    collected_text_data = image_to_text_lines(image)
    with open(text_file_name, 'w', encoding='utf-8') as fp:
        fp.write(u'\n'.join(collected_text_data))
        # import pdb
        # pdb.set_trace()
        # print(u'\n'.join(collected_text_data))
        print('Image2Text {} {}'.format(image_filename, text_file_name))


def async_main():
    import asyncio
    import random
    dict = {'router1': {'id': 1, 'name': 'rtr1_core'},
            'router2': {'id': 2, 'name': 'rt2_core'},
            'router3': {'id': 3, 'name': 'rtr3_access'}}

    async def process_rtr(id, name):
        """Do your execution here."""
        s_time = await asyncio.sleep(random.randint(0, 5))
        print(f"Processing {id}, {name}")

    loop = asyncio.get_event_loop()
    tasks = [asyncio.ensure_future(process_rtr(**router_details))
             for router, router_details
             in dict.items()]
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


if __name__ == '__main__':
    from glob import glob

    for image_filename in glob(config.BINARIZE_ROOT_DIR + '/*'):
        main(image_filename)
        # break
