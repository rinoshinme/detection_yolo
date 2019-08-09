import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def ckpt_inspect(ckpt_file, output_file=None):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # for op_name in var_to_shape_map:
    #     print(op_name, var_to_shape_map[op_name])
    for op_name in sorted(var_to_shape_map.keys()):
        print(op_name, var_to_shape_map[op_name])


def graph_inspect(session, output_file=None):
    graph = session.graph
    for node in graph.as_graph_def().node:
        print(node.name)
