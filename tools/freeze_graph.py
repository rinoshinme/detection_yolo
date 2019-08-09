import tensorflow as tf
from yolo.yolov3 import YOLOv3


def freeze_graph(ckpt_path, pb_path, output_nodes):
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    model = YOLOv3(input_data, is_training=False)
    print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                       input_graph_def=sess.graph.as_graph_def(),
                                                                       output_node_names=output_nodes)

    with tf.gfile.GFile(pb_path, "wb") as f:
        f.write(converted_graph_def.SerializeToString())


if __name__ == '__main__':
    pb_file = "./yolov3_coco.pb"
    ckpt_file = "./checkpoint/yolov3_coco_demo.ckpt"
    output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]
    freeze_graph(ckpt_file, pb_file, output_node_names)
