import json
import xml.dom.minidom
import xmltodict


def single_annotated_box_to_xml(annotation):
    _object_xml_format = dict({'name': annotation['tag'],
                               'pose': 'Unspecified',
                               'truncated': '0',
                               'difficult': '0',
                               'bndbox': {
                                   'xmin': annotation['x'],
                                   'ymin': annotation['y'],
                                   'xmax': annotation['x'] + annotation['width'],
                                   'ymax': annotation['y'] + annotation['height'], }})
    return _object_xml_format


def annotated_boxes_to_xml(annotations):
    objects = []
    for each in annotations:
        objects.append(single_annotated_box_to_xml(each))
    return objects


def validate_tags_and_regions(form_data):
    sent_info = json.loads(form_data['sendInfo'])
    # sent_info.keys() => dict_keys(['url', 'folder', 'id', 'width', 'height', 'annotations'])
    default_xml_format = {
        'annotation': {
            'folder': sent_info['folder'],
            'filename': sent_info['id'],
            'path': sent_info['url'],
            'source': {
                'database': 'Unknown'
            },
            'size_part': {
                'width': sent_info['width'],
                'height': sent_info['height'],
                'depth': '3'},
            'segmented': '0',
            'object': annotated_boxes_to_xml(sent_info['annotations'])
        }
    }
    xml_string = xmltodict.unparse(default_xml_format)
    xml_string = xml.dom.minidom.parseString(xml_string)
    pretty_xml_as_string = xml_string.toprettyxml()
    print(pretty_xml_as_string)


def read_xml_annotated_file(filename):
    return None