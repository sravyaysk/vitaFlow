import os

ROOT_DIR = os.path.dirname(__file__)

# Image path to be used in the HTML client
# IMAGE_WEB_DIR = "data/images"

# Image path for internal PHP use
IMAGE_ROOT_DIR = "static/data/images"

# To store cropped images - original images
CROPPER_ROOT_DIR = "static/data/cropper"

# To store annotation xml files
ANNOTATIONS_DIR = "static/data/annotations"

# Collection name
COLLECTION_NAME = "collection_01"

# Not annotated image 80% to be presented to user
ratio_new_old = 80



# Create directories

for each_dir in [
    IMAGE_ROOT_DIR,
    CROPPER_ROOT_DIR,
    ANNOTATIONS_DIR
    ]:
    each_dir = os.path.join(ROOT_DIR, each_dir)
    if not os.path.isdir(each_dir):
        print('Created missing directory {}!'.format(each_dir))
        os.mkdir(each_dir)
    else:
        print('Directory avail at {}!'.format(each_dir))


