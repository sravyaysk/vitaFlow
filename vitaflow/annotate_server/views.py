import flask
import glob
import os
import random

try:
    from . import config
except:
    import config

PWD = os.path.dirname(config.__file__)


class GetNewImage:
    ImageFiles = {}
    XmlFiles = {}
    PendingImage = []

    @staticmethod
    def refresh():
        # with timely invervals - 5 mins update
        GetNewImage.XmlFiles = GetNewImage._get_annotations()
        GetNewImage.ImageFiles = GetNewImage._get_images()
        GetNewImage.PendingImage = list(set(GetNewImage.ImageFiles)
                                        - set(GetNewImage.XmlFiles))

    @staticmethod
    def update_data(data):
        pass

    @staticmethod
    def get_new_image(data):
        if data:
            GetNewImage.update_data(data)
        image_file = random.choice(list(GetNewImage.XmlFiles.keys()))
        send_info = {
            'id': image_file,
            'url': GetNewImage.XmlFiles[image_file]['url'],
            'folder': '',
            "annotations": []
        }
        return send_info

    @staticmethod
    def get_old_image(data):
        image_file = random.choice(list(GetNewImage.XmlFiles.keys()))
        send_info = {
            'id': image_file,
            'url': GetNewImage.XmlFiles[image_file]['url'],
            'folder': '',
            "annotations": []
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
            url = each.split(PWD)[-1].lstrip('/')
            file = os.path.basename(url)
            # print(url, file)
            images_dict[file] = {
                'url': url,
                'fullpath': each
            }
        return images_dict

    @staticmethod
    def _get_annotations():
        xml_dict = {}
        xml_files = GetNewImage._parser_folder('static/data/annotations/', ['.xml'])
        for each in xml_files:
            url = each.split(PWD)[-1].lstrip('/')
            file = os.path.basename(url)
            # print(url, file)
            xml_dict[file] = {
                'url': url,
                'fullpath': each
            }
        return xml_dict

