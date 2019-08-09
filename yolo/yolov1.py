import tensorflow as tf
import numpy as np
from utils.layers import leaky_relu
from config.yolov1_config import cfg
slim = tf.contrib.slim


class YOLOv1(object):
    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size * self.cell_size) * \
                           (self.num_classes + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = 0
        self.boundary2 = 0

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA
        self.is_training = is_training

    def build_network(self, images, num_output, keep_prob=0.5):
        with tf.variable_scope('yolo'):
            net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
            net = tf.layers.conv2d(net, 64, [3, 3], [2, 2], padding='valid',
                                   activation=leaky_relu(self.alpha),
                                   name='conv_2')
        return net

    def loss_layer(self):
        pass


if __name__ == '__main__':
    model = YOLOv1()
