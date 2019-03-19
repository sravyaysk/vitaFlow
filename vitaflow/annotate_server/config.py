import os

from bin.utils import check_n_create

ROOT_DIR = os.path.dirname(__file__)

# Image path to be used in the HTML client
# IMAGE_WEB_DIR = "data/images"

# Image path for internal PHP use
IMAGE_ROOT_DIR = "static/data/images"

# To store cropped images - original images
CROPPER_ROOT_DIR = "static/data/cropper"

# To store cropped images - original images
BINARIZE_ROOT_DIR = "static/data/binarisation"

# To store annotation xml files
ANNOTATIONS_DIR = "static/data/annotations"

# Collection name
# COLLECTION_NAME = "collection_01"

# Not annotated image 80% to be presented to user
ratio_new_old = 80

# Acceptable file extension
IMAGE_EXTS = ['.jpg']

# Time inverval to re-check images
OS_FILE_REFRESH_TIME_INVTERVAL = 2 * 60  # Shift to inotify

# create missing dir
for each_dir in [IMAGE_ROOT_DIR,
                 CROPPER_ROOT_DIR,
                 BINARIZE_ROOT_DIR,
                 ANNOTATIONS_DIR]:
    each_dir = os.path.join(ROOT_DIR, each_dir)
    check_n_create(each_dir)
