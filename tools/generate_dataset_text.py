"""
Generate dataset text file from Pascal VOC annotations
"""
import os
import sys
import xml.etree.ElementTree as ET
import platform
cur_path = os.path.abspath(__file__)
sys.path.append(os.path.join('../', cur_path))
from utils.meta import read_class_names


os_name = platform.system()

if os_name == 'Windows':
    VOCDEVKIT_PATH = r'E:\dataset1\VOCdevkit'
elif os_name == 'linux':
    VOCDEVKIT_PATH = '/home/deploy/rinoshinme/data/VOCdevkit'
else:
    raise ValueError("system not specified.")

IMAGE_PATH = os.path.join(VOCDEVKIT_PATH, 'VOC2007', 'JPEGImages')
ANNOTATION_PATH = os.path.join(VOCDEVKIT_PATH, 'VOC2007', 'Annotations')
TEXT_PATH = os.path.join(VOCDEVKIT_PATH, 'VOC2007', 'ImageSets', 'Main')


def generate_dataset(mode, save_path, class_text):
    assert mode in ['train', 'val', 'trainval', 'test']
    id2name = read_class_names(class_text)
    name2id = {val: key for key, val in id2name.items()}

    index_text = os.path.join(TEXT_PATH, mode + '.txt')
    save_fp = open(save_path, 'w')

    with open(index_text, 'r') as f:
        for line in f:
            index_str = line.strip()
            img_path = os.path.join(IMAGE_PATH, index_str + '.jpg')
            annotation_path = os.path.join(ANNOTATION_PATH, index_str + '.xml')

            bboxes = []
            # parse xml file
            tree = ET.parse(annotation_path)
            objs = tree.findall('object')
            for obj in objs:
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                name = obj.find('name').text.lower().strip()
                label = name2id[name]
                bboxes.append((x1, y1, x2, y2, label))

            # write as a single line
            save_fp.write('%s' % img_path)
            for bbox in bboxes:
                save_fp.write(' %d,%d,%d,%d,%d' % (bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]))
            save_fp.write('\n')


if __name__ == '__main__':
    savepath = r'../config/voc2007_{}.txt'
    classtext = r'../config/voc.names'
    for phase in ['train', 'trainval', 'test']:
        generate_dataset(phase, savepath.format(phase), classtext)
