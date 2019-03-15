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

ALL_MERCHANTS_NAMES = '''
Publix
Wegmans
Trader Joe's
H-E-B
Aldi
Harris Teeter
Hy-Vee
Costco
WinCo
Whole Foods
Fry's
Kroger
Target
Winn-Dixie
ShopRite
Food Lion
Albertsons
Meijer
Sam's Club
Giant Food
Safeway
Stop & Shop
Walmart
'''.strip().splitlines()

ALL_LINE_ITEMS = '''
All-purpose flour
American cheese
Apples 
Banana 
Beef Round
Boneless chicken breast
Boneless pork chop
Broccoli
Chicken Breasts
Chocolate chip cookies
Creamy peanut butter
Dried beans
Eggs (regular) 
Frozen turkey
Ground beef
Ice cream
Lemons
Lettuce 
Loaf of Fresh White Bread 
Local Cheese
Milk (regular)
Navel oranges
Onion 
Oranges 
Pasta
Potato 
Rice (white)
Salted butter
Seedless grapes
Sirloin steak
Sliced bacon
Strawberries
Sugar
Tomato 
Top round steak
Wheat bread
'''.strip().splitlines()

ALL_LINE_ITEMS = [_.strip() for _ in ALL_LINE_ITEMS]
ALL_MERCHANTS_NAMES = [_.strip() for _ in ALL_MERCHANTS_NAMES]


def get_random_merchant_name():
    return random.choice(ALL_MERCHANTS_NAMES)


def get_random_line_item():
    return random.choice(ALL_LINE_ITEMS)


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
    receipt_text = []
    _number_of_line_items = random.choice(range(5, 15))

    image = Image.new(mode="RGB", size=(220, 250 + _number_of_line_items * 15), color=(255, 255, 255))
    # initialise the drawing context with
    # the image object as background
    draw = ImageDraw.Draw(image)

    _merchant_tag = get_random_merchant_name().center(10)
    _date = random.random()
    date, _, _ = get_random_date("1/1/2015 1:30 PM", "1/1/2019 4:50 AM", _date)
    date = "Date : " + str(date)
    _invoice_text = "Invoice : " + str(get_random_number(min_num_chars=4, max_num_chars=6))

    receipt_text = [
        _merchant_tag.strip(),
        _invoice_text.strip() + ' ' + date,
        "Item  Price",
    ]
    draw = insert_text(draw=draw, x=60, y=30, text=_merchant_tag, font_size=20)
    draw = insert_text(draw=draw, x=20, y=80, text=_invoice_text, font_size=10)
    draw = insert_text(draw=draw, x=120, y=80, text=date, font_size=10)
    draw = insert_text(draw=draw, x=30, y=120, text="Item", font_size=10)
    draw = insert_text(draw=draw, x=150, y=120, text="Price", font_size=10)

    total = 0
    total_y = -1
    _min_line_width = 15
    for i in range(1, _number_of_line_items):
        # item
        item_text = get_random_line_item().center(10)
        draw = insert_text(draw=draw, x=20, y=140 + (i * _min_line_width), text=item_text, font_size=10)

        # item - price
        item_price = get_random_number(max_num_chars=5)
        draw = insert_text(draw=draw, x=150, y=140 + (i * _min_line_width), text=item_price, font_size=10)
        receipt_text.append('{}  {}'.format(item_text.strip(), item_price).strip())

        total = total + item_price

    total_y = 140 + (i * _min_line_width)

    total_y += 10
    # draw = insert_text(draw=draw, x=20, y=total_y, text="-" * 45 , font_size=10)
    total_y += 20
    _text = "Tax : " + str(0.15)
    receipt_text.append(_text)
    draw = insert_text(draw=draw, x=125, y=total_y, text=_text, font_size=10)
    total_y += 20
    _text = "Sub Total : " + str(total)
    receipt_text.append(_text)
    draw = insert_text(draw=draw, x=100, y=total_y, text=_text, font_size=10)
    total_y += 20
    _text = "Total : " + str(total + total * 0.15)
    receipt_text.append(_text)
    draw = insert_text(draw=draw, x=120, y=total_y, text=_text, font_size=10)

    # write to a file
    fp = open(file_path + '.txt', 'w')
    try:
        fp.writelines(['{}\n'.format(_) for _ in receipt_text])
    except UnicodeEncodeError:
        import pdb
        pdb.set_trace()
    # save image
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
    number_files = 100
    os.makedirs(OUT_DIR + "/train/")

    for i in tqdm(range(number_files)):
        create_naive_receipt(OUT_DIR + "/train/" + str(i) + ".jpg")
        # replicate_xml(out_file_path=OUT_DIR + "/train/" + str(i) + ".xml")


def test():
    number_files = 100
    os.makedirs(OUT_DIR + "/test/")

    for i in tqdm(range(number_files)):
        create_naive_receipt(OUT_DIR + "/test/" + str(i) + ".jpg")
        replicate_xml(out_file_path=OUT_DIR + "/test/" + str(i) + ".xml")


def eval():
    number_files = 1
    os.makedirs(OUT_DIR + "/eval/")

    for i in tqdm(range(number_files)):
        create_naive_receipt(OUT_DIR + "/eval/" + str(i) + ".jpg")
        replicate_xml(out_file_path=OUT_DIR + "/eval/" + str(i) + ".xml")


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)
    train()
    # test()
    # eval()


if __name__ == '__main__':
    main()
