#! python

import xml.etree.ElementTree
from glob import glob

import config
import pandas as pd

bag = [
    # (filename, tag)
]

for annotation_file in glob('{}/*.xml'.format(config.ANNOTATIONS_DIR)):
    root = xml.etree.ElementTree.parse(annotation_file).getroot()
    for each_object in root.findall('object'):
        name = each_object.find('name').text
        bag.append((annotation_file, name))

df = pd.DataFrame(bag, columns=['filename', 'tag'])
print('Loc: /var/www/html/images_annotation/data/annotations')
print('Total Number of files processed:', len(df.filename.unique()))
print(df.tag.value_counts())
