from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
from glob import glob

images_src = '/Users/sampathm/Desktop/test_images/*'
images_dest = '/Users/sampathm/Desktop/test_images2/'


def image_annotations(images_src=images_src, images_dest=images_dest):
    path_to_tensorflow_model = '/Users/sampathm/devbox/object_detector_app/inference_graph/frozen_inference_graph.pb'
    category_index = \
        {1: {'id': 1, 'name': 'merchant'},
         2: {'id': 2, 'name': 'receipt_number'},
         3: {'id': 3, 'name': 'date'},
         4: {'id': 4, 'name': 'line_items_and_value'},
         5: {'id': 5, 'name': 'total'},
         6: {'id': 6, 'name': 'tax'},
         7: {'id': 7, 'name': 'mode_of_payment'}}

    def get_box_dims(box, image_shape):
        ymin, xmin, ymax, xmax = box
        im_width, im_height, im_depth = image_shape
        ymin, xmin, ymax, xmax = map(int, (xmin * im_width, xmax * im_width,
                                           ymin * im_height, ymax * im_height))
        return (ymax, xmax, ymin, xmin)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_tensorflow_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    bag = []

    for image_path in glob(images_src):
        image = plt.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        mask = scores > 0.3

        image_shape = image.shape

        coords = list(map(lambda x: get_box_dims(x, image_shape), boxes[mask].tolist()))
        tags = list(map(lambda x: category_index[int(x)]['name'], classes[mask].tolist()))
        scores = scores[mask].tolist()
        bag.append({'image_loc': image_path,
                    'dest': images_dest,
                    'coords': coords,
                    'tags': tags,
                    'scores': scores})

    # pprint(bag)
    return bag


if __name__ == '__main__':
    # pprint(image_annotations())
    from image_processing import ImageToTextImages

    for each in image_annotations(images_src, images_dest):
        t = ImageToTextImages(each['image_loc'], each['dest'])
        t.multiple_crop_and_save(each['coords'], each['tags'])
