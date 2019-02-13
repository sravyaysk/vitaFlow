import glob
import os
import random
import time

try:
    from . import config
except:
    import config

PWD = os.path.dirname(config.__file__)

def trim_ext(x):
    return x.rsplit('.')[-2]


class GetNewImage:
    # TODO: Naming Convention to changes to lower case.
    ImageFiles = {}
    XmlFiles = {}
    PendingImages = []
    CompletedImages = []
    _last_refresh_ = 0
    # TODO: put below param in config.py
    _refresh_interval_ = 2 * 60

    @staticmethod
    def refresh(image=None):
        # only once in - 5 mins update.
        if image:
            GetNewImage.update_data(image)
        if time.time() - GetNewImage._last_refresh_ < GetNewImage._refresh_interval_:
            print('GetNewImage is still fresh!!')
            return

        GetNewImage._last_refresh_ = time.time()
        print('GetNewImage is refreshed!!')
        GetNewImage.XmlFiles = GetNewImage._get_annotations()
        GetNewImage.ImageFiles = GetNewImage._get_images()

        xml_files_keys = set(GetNewImage.XmlFiles.keys())
        image_files_keys = set(GetNewImage.ImageFiles.keys())
        GetNewImage.PendingImages = list(image_files_keys - xml_files_keys)
        GetNewImage.CompletedImages = list(image_files_keys.intersection(xml_files_keys))

    @staticmethod
    def update_data(image=None):
        image_key = trim_ext(image)
        if image_key in GetNewImage.PendingImages:
            GetNewImage.PendingImages.remove(image_key)
            GetNewImage.CompletedImages.append(image_key)
            print('Image Annotations Done for {}'.format(image))

    @staticmethod
    def get_new_image():
        if GetNewImage.PendingImages:
            image_file = random.choice(GetNewImage.PendingImages)
            # image_file = [each for each in GetNewImage.ImageFiles.keys() if each.startswith(image_file)][0]
            send_info = {
                'id': GetNewImage.ImageFiles[image_file]['file'],
                'url': GetNewImage.ImageFiles[image_file]['url'],
                'folder': '',
                "annotations": []
            }
        else:
            # TODO: Add a default Image for display in case no images are pending.
            send_info = {"url": "/static/data/images/pexels-photo-60091.jpg",
                         "id": "pexels-photo-60091.jpg",
                         "folder": "collection_01/part_1",
                         "annotations": [
                             {
                                 "tag": "Eagle",
                                 "x": 475, "y": 225,
                                 "width": 230.555555554,
                                 "height": 438.888888886}
                         ]
                         }
        print('GetNewImage returned {}'.format(send_info['id']))
        return send_info

    @staticmethod
    def get_old_image():
        if GetNewImage.XmlFiles:
            image_file = random.choice(list(GetNewImage.XmlFiles.keys()))
            send_info = {
                'id': image_file,
                'url': GetNewImage.XmlFiles[image_file]['url'],
                'folder': '',
                "annotations": []
            }
        else:
            # TODO: Add a default Image for display in case no images are annotated.
            send_info = {"url": "/static/data/images/pexels-photo-60091.jpg",
                         "id": "pexels-photo-60091.jpg",
                         "folder": "collection_01/part_1",
                         "annotations": [
                             {
                                 "tag": "Eagle",
                                 "x": 475, "y": 225,
                                 "width": 230.555555554,
                                 "height": 438.888888886}
                         ]
                         }
        return send_info

    @staticmethod
    def _parser_folder(folder='static/data/images', exts=None):
        search_folder = folder
        if os.path.isdir(os.path.join(PWD, folder)):
            search_folder = os.path.join(PWD, folder)
        # print('Request Parsing: {}'.format(search_folder))
        bag = []
        for filename in glob.iglob(search_folder + '*', recursive=True):
            file = os.path.basename(filename)
            url = os.path.join(folder, file)
            bag.append(url)
        if not exts:
            return bag
        _bag = []
        for ext in exts:
            for file in bag:
                if file.endswith(ext):
                    _bag.append(file)
        return _bag

    @staticmethod
    def _get_images():
        images_dict = {}
        for each in GetNewImage._parser_folder('static/data/images/'):
            url = each.split(PWD)[-1].lstrip(os.sep)
            file = os.path.basename(url)
            # print(url, file)
            images_dict[trim_ext(file)] = {
                'url': url,
                'fullpath': each,
                'file': file
            }
        return images_dict

    @staticmethod
    def _get_annotations():
        xml_dict = {}
        xml_files = GetNewImage._parser_folder('static/data/annotations/', ['.xml'])
        for each in xml_files:
            url = each.split(PWD)[-1].lstrip(os.sep)
            file = os.path.basename(url)
            # print(url, file)
            xml_dict[trim_ext(file)] = {
                'url': url,
                'fullpath': each,
                'file': file
            }
        return xml_dict
