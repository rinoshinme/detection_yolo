import tensorflow as tf
import numpy as np
from config.yolov3_config import cfg
from yolo.darknet53 import darknet53
import utils.layers
from utils.boxes import bbox_giou, bbox_iou
from utils.losses import focal
from utils.meta import get_anchors, get_shape,read_class_names
from utils.model import ckpt_inspect, graph_inspect
from tensorflow.python import pywrap_tensorflow
import time


class YOLOv3(object):
    def __init__(self, input_data, is_training):
        self.is_training = is_training
        self.classes = read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD

        # self._build_network(input_data)
        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self._build_network(input_data)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def _build_network(self, input_data):
        route_1, route_2, input_data = darknet53(input_data, self.is_training)
        # route_1: stride=8
        # route_2: stride=16
        # input_data: stride=32

        input_data = utils.layers.convolutional(input_data, (1, 1, 1024, 512), self.is_training, name='conv52')
        input_data = utils.layers.convolutional(input_data, (3, 3, 512, 1024), self.is_training, name='conv53')
        input_data = utils.layers.convolutional(input_data, (1, 1, 1024, 512), self.is_training, name='conv54')
        input_data = utils.layers.convolutional(input_data, (3, 3, 512, 1024), self.is_training, name='conv55')
        input_data = utils.layers.convolutional(input_data, (1, 1, 1024, 512), self.is_training, name='conv56')

        conv_lobj_branch = utils.layers.convolutional(input_data, (3, 3, 512, 1024),
                                                      self.is_training, name='conv_lobj_branch')
        conv_lbbox = utils.layers.convolutional(conv_lobj_branch,
                                                (1, 1, 1024, self.anchor_per_scale * (self.num_classes + 5)),
                                                is_training=self.is_training, name='conv_lbbox',
                                                activate=False, bn=False)

        input_data = utils.layers.convolutional(input_data, (1, 1, 512, 256), self.is_training, 'conv57')
        input_data = utils.layers.upsample(input_data, name='updample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = utils.layers.convolutional(input_data, (1, 1, 768, 256), self.is_training, 'conv58')
        input_data = utils.layers.convolutional(input_data, (3, 3, 256, 512), self.is_training, 'conv59')
        input_data = utils.layers.convolutional(input_data, (1, 1, 512, 256), self.is_training, 'conv60')
        input_data = utils.layers.convolutional(input_data, (3, 3, 256, 512), self.is_training, 'conv61')
        input_data = utils.layers.convolutional(input_data, (1, 1, 512, 256), self.is_training, 'conv62')

        conv_mobj_branch = utils.layers.convolutional(input_data, (3, 3, 256, 512),
                                                      self.is_training, name='conv_mobj_branch')
        conv_mbbox = utils.layers.convolutional(conv_mobj_branch,
                                                (1, 1, 512, self.anchor_per_scale * (self.num_classes + 5)),
                                                is_training=self.is_training, name='conv_mbbox',
                                                activate=False, bn=False)

        input_data = utils.layers.convolutional(input_data, (1, 1, 256, 128), self.is_training, 'conv63')
        input_data = utils.layers.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = utils.layers.convolutional(input_data, (1, 1, 384, 128), self.is_training, 'conv64')
        input_data = utils.layers.convolutional(input_data, (3, 3, 128, 256), self.is_training, 'conv65')
        input_data = utils.layers.convolutional(input_data, (1, 1, 256, 128), self.is_training, 'conv66')
        input_data = utils.layers.convolutional(input_data, (3, 3, 128, 256), self.is_training, 'conv67')
        input_data = utils.layers.convolutional(input_data, (1, 1, 256, 128), self.is_training, 'conv68')

        conv_sobj_branch = utils.layers.convolutional(input_data, (3, 3, 128, 256),
                                                      self.is_training, name='conv_sobj_branch')
        conv_sbbox = utils.layers.convolutional(conv_sobj_branch,
                                                (1, 1, 256, self.anchor_per_scale * (self.num_classes + 5)),
                                                is_training=self.is_training, name='conv_sbbox',
                                                activate=False, bn=False)

        # strides for [s, m, l] are [8, 16, 32]
        return conv_lbbox, conv_mbbox, conv_sbbox

    def decode(self, conv_output, anchors, stride):
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_classes))
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[..., 2:4]
        conv_raw_conf = conv_output[..., 4:5]
        conv_raw_prob = conv_output[..., 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        # calculate loss per feature layer
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_classes))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 5:]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
        iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = focal(respond_bbox, pred_conf)
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf) +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):
        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss

    @staticmethod
    def load_trained_weights(session, weight_path):
        # print('---------------------------weight_path')
        # ckpt_inspect(weight_path)
        # print('---------------------------graph_nodes')
        # graph_inspect(session)

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(session, weight_path)

    @staticmethod
    def load_initial_weights(session, weight_path, train_layers=None):
        if train_layers is None:
            train_layers = []

        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print('{}: Start Loading weights...'.format(start_time))
        reader = pywrap_tensorflow.NewCheckpointReader(weight_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for op_name in var_to_shape_map:
            if op_name == 'global_step':
                continue

            op_name_list = op_name.split('/')
            union_list = [item for item in op_name_list if item in train_layers]
            if len(union_list) != 0:
                continue

            try:
                with tf.variable_scope('/'.join(op_name.split('/')[0:-1]), reuse=True):
                    data = reader.get_tensor(op_name)
                    var = tf.get_variable(op_name.split('/')[-1], trainable=False)
                    # var = tf.get_variable(op_name.split('/')[-1], trainable=True)
                    session.run(var.assign(data))
            except ValueError:
                tmp1 = list(op_name in str(item) for item in tf.global_variables())
                tmp2 = np.sum([int(item) for item in tmp1])
                if tmp2 == 0:
                    print("Don't be loaded: {}, cause: {}".format(op_name, "new model no need this variable."))
                else:
                    print("Don't be loaded: {}, cause: {}".format(op_name, ValueError))
        end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print('{}: Loading parameters finished...'.format(end_time))


if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, shape=[None, 416, 416, 3], name='inputs')
    model = YOLOv3(input_tensor, is_training=True)
