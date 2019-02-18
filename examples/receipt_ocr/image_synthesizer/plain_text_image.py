# https://github.com/google/fonts

import base64
import os
import random
import shutil
import string
import time
import xml.etree.ElementTree as ET

import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

OUT_DIR = "receipt_mock_data"


def get_random_string(max_num_chars=15):
    num_chars = random.randint(5, max_num_chars)
    # digits = "".join( [random.choice(string.digits) for i in range(8)] )
    chars = "".join([random.choice(string.ascii_letters) for i in range(num_chars)])
    return chars


def get_random_number(min_num_chars=1, max_num_chars=5):
    num_chars = random.randint(min_num_chars, max_num_chars)
    digits = "".join([random.choice(string.digits) for i in range(num_chars)])
    return int(float(digits))


def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def get_random_date(start, end, prop):
    text = strTimeProp(start, end, '%m/%d/%Y %I:%M %p', prop)
    date, time, am_pm = text.split(" ")
    return date, time, am_pm


def read_from_git(url):
    url = "https://github.com/google/fonts/blob/master/apache/roboto/Roboto-Bold.ttf"
    req = requests.get(url)
    if req.status_code == requests.codes.ok:
        req = req.json()  # the response is a JSON
        # req is now a dict with keys: name, encoding, url, size ...
        # and content. But it is encoded with base64.
        content = base64.decodebytes(req['content'])
        return content
    else:
        print('Content was not found.')


def insert_text(draw: ImageDraw,
                x,
                y,
                text,
                color='rgb(0, 0, 0)',
                font_file='fonts/Roboto-Bold.ttf',
                font_size=12):
    text = str(text)
    font = ImageFont.truetype(font_file, size=font_size)
    draw.text((x, y), text, fill=color, font=font)
    return draw


def create_naive_receipt(file_path):
    # create Image object with the input image
    image = Image.new(mode="RGB", size=(220, 300), color=(255, 255, 255))
    # initialise the drawing context with
    # the image object as background
    draw = ImageDraw.Draw(image)
    draw = insert_text(draw=draw, x=60, y=30, text=get_random_string(max_num_chars=10), font_size=20)
    date, _, _ = get_random_date("1/1/2015 1:30 PM", "1/1/2019 4:50 AM", random.random())
    draw = insert_text(draw=draw, x=20, y=80,
                       text="Invoice : " + str(get_random_number(min_num_chars=4, max_num_chars=6)), font_size=10)
    draw = insert_text(draw=draw, x=120, y=80, text="Date : " + date, font_size=10)
    draw = insert_text(draw=draw, x=30, y=120, text="Item", font_size=10)
    draw = insert_text(draw=draw, x=150, y=120, text="Price", font_size=10)

    for i in range(1, 5):
        x = 20
        y = 140 + (i * 15)
        text = get_random_string(max_num_chars=10)
        draw = insert_text(draw=draw, x=x, y=y, text=text, font_size=10)

    total = 0
    total_y = -1
    for i in range(1, 5):
        x = 150
        y = 140 + (i * 15)
        text = get_random_number(max_num_chars=5)
        draw = insert_text(draw=draw, x=x, y=y, text=text, font_size=10)
        total = total + text
        total_y = y

    draw = insert_text(draw=draw, x=120, y=total_y + 20, text="Total : " + str(total), font_size=10)
    image.save(file_path, "JPEG")


def replicate_xml(out_file_path, image_store_path="images", in_file_pathh="0.xml"):
    tree = ET.parse(in_file_pathh)
    root = tree.getroot()
    for elem in root.iter('filename'):
        elem.text = out_file_path.split("/")[-1]
    for elem in root.iter('path'):
        elem.text = image_store_path + "/" + out_file_path.split("/")[-1]
    tree.write(out_file_path, encoding='utf-8', xml_declaration=True)


def train():
    number_files = 2000
    os.makedirs(OUT_DIR + "/train/")

    for i in tqdm(range(number_files)):
        create_naive_receipt(OUT_DIR + "/train/" + str(i) + ".jpg")
        replicate_xml(out_file_path=OUT_DIR + "/train/" + str(i) + ".xml")


def test():
    number_files = 400
    os.makedirs(OUT_DIR + "/test/")

    for i in tqdm(range(number_files)):
        create_naive_receipt(OUT_DIR + "/test/" + str(i) + ".jpg")
        replicate_xml(out_file_path=OUT_DIR + "/test/" + str(i) + ".xml")


def eval():
    number_files = 100
    os.makedirs(OUT_DIR + "/eval/")

    for i in tqdm(range(number_files)):
        create_naive_receipt(OUT_DIR + "/eval/" + str(i) + ".jpg")
        replicate_xml(out_file_path=OUT_DIR + "/eval/" + str(i) + ".xml")


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)
    train()
    test()
    # eval()


if __name__ == '__main__':
    main()
