"""
Test YOLOv3 using pb file
"""
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from utils.model import read_pb_return_tensors
from utils.processing import image_preprocess, postprocess_boxes, draw_bbox
from utils.bboxes_np import nms


class TestYOLOv3(object):
    def __init__(self, pb_file):
        self.elements = ["input/input_data:0",
                         "pred_sbbox/concat_2:0",
                         "pred_mbbox/concat_2:0",
                         "pred_lbbox/concat_2:0"]
        self.session = tf.Session()
        self.graph = self.session.graph
        self.tensors = read_pb_return_tensors(self.graph, pb_file, self.elements)
        self.input_size = 416
        self.num_classes = 20

    def test_image(self, image_path):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = image_preprocess(np.copy(original_image), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.session.run(
            [self.tensors[1], self.tensors[2], self.tensors[3]],
            feed_dict={self.tensors[0]: image_data}
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image_size, self.input_size, 0.3)
        bboxes = nms(bboxes, 0.45, method='nms')

        image = draw_bbox(original_image, bboxes)
        image = Image.fromarray(image)
        image.show()
