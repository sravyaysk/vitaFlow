import os
import random
import time

import config
from bin.datatypes import Singleton
from bin.utils import get_folder_config
from bin.utils import trim_file_ext


# config.IMAGE_ROOT_DIR = os.path.dirname(config.__file__)

# noinspection PyCompatibility
class GetNewImage(metaclass=Singleton):
    # input images
    receipt_images = {}
    # features - cropping & rotation
    cropper_images = {}
    # feature - bin_images
    bin_images = {}
    # output files
    annotated_files = {}
    # other
    pending_images = []
    completed_images = []
    _last_refresh_ = 0
    _refresh_interval_ = config.OS_FILE_REFRESH_TIME_INVTERVAL
    _default_image_ = "/static/images/NoImage.png"  # TODO: Remove this - set this at UI - JS Level

    @staticmethod
    def get_default_request_response():
        send_info = {"url": GetNewImage._default_image_,
                     "id": os.path.basename(GetNewImage._default_image_),
                     "folder": "",
                     "annotations": []}
        return send_info

    @staticmethod
    def get_specific_image(image_file):
        image_name = trim_file_ext(image_file)
        send_info = {
            'id': GetNewImage.receipt_images[image_name]['file'],
            'url': GetNewImage.receipt_images[image_name]['url'],
            'folder': '',
            "annotations": []
        }
        return send_info

    @staticmethod
    def get_old_image():
        if GetNewImage.annotated_files:
            image_file = random.choice(list(GetNewImage.annotated_files.keys()))
            send_info = GetNewImage.get_specific_image(image_file)
            # TODO - Add Annotations
        else:
            send_info = GetNewImage.get_default_request_response()
        return send_info

    @staticmethod
    def get_new_image():
        if GetNewImage.pending_images:
            image_file = random.choice(GetNewImage.pending_images)
            send_info = GetNewImage.get_specific_image(image_file)
        else:
            send_info = GetNewImage.get_default_request_response()
        print('GetNewImage returned {}'.format(send_info['id']))
        return send_info

    @staticmethod
    def _get_images():
        images_dict = get_folder_config(os.path.join(config.ROOT_DIR, config.IMAGE_ROOT_DIR),
                                        config.IMAGE_EXTS,
                                        config.ROOT_DIR)
        return images_dict

    @staticmethod
    def _get_annotations():
        xml_dict = get_folder_config(os.path.join(config.ROOT_DIR, config.ANNOTATIONS_DIR),
                                     ['.xml'],
                                     config.ROOT_DIR)
        return xml_dict

    @staticmethod
    def update_data(image=None):
        image_key = trim_file_ext(image)
        if image_key in GetNewImage.pending_images:
            GetNewImage.pending_images.remove(image_key)
            GetNewImage.completed_images.append(image_key)
            del GetNewImage.receipt_images[image_key]
            print('Image Annotations Done for {}'.format(image))

    @staticmethod
    def refresh(image=None):
        if image:
            GetNewImage.update_data(image)
        if time.time() - GetNewImage._last_refresh_ < GetNewImage._refresh_interval_:
            print('GetNewImage is still fresh!!')
            return

        GetNewImage._last_refresh_ = time.time()
        print('GetNewImage is refreshed!!')
        GetNewImage.annotated_files = GetNewImage._get_annotations()
        GetNewImage.receipt_images = GetNewImage._get_images()

        xml_files_keys = set(GetNewImage.annotated_files.keys())
        image_files_keys = set(GetNewImage.receipt_images.keys())
        GetNewImage.pending_images = list(image_files_keys - xml_files_keys)
        GetNewImage.completed_images = list(image_files_keys.intersection(xml_files_keys))
