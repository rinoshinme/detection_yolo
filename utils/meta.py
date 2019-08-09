import numpy as np
import tensorflow as tf


def get_shape(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor.shape
    elif isinstance(tensor, tf.Tensor):
        return tensor.get_shape().as_list()
    else:
        return 'shape not supported'


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    # 3 feature layers, 3 anchor scales for each feature layer
    # (width, height) for each anchor
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)



