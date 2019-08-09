import tensorflow as tf
import utils.layers


def darknet53(input_data, is_training):
    with tf.variable_scope('darknet-53'):
        input_data = utils.layers.convolutional(input_data, filters_shape=(3, 3, 3, 32),
                                                is_training=is_training, name='conv0')
        input_data = utils.layers.convolutional(input_data, filters_shape=(3, 3, 32, 64),
                                                is_training=is_training, name='conv1', downsample=True)

        for i in range(1):
            input_data = utils.layers.residual_block(input_data, 64, 32, 64,
                                                     is_training=is_training, name='residual{}'.format(i + 0))

        # down-sample
        input_data = utils.layers.convolutional(input_data, filters_shape=(3, 3, 64, 128),
                                                is_training=is_training, name='conv4', downsample=True)

        for i in range(2):
            input_data = utils.layers.residual_block(input_data, 128, 64, 128,
                                                     is_training=is_training, name='residual{}'.format(i + 1))
        # down-sample
        input_data = utils.layers.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                                is_training=is_training, name='conv9', downsample=True)

        for i in range(8):
            input_data = utils.layers.residual_block(input_data, 256, 128, 256,
                                                     is_training=is_training, name='residual{}'.format(i + 3))
        route_1 = input_data
        input_data = utils.layers.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                                is_training=is_training, name='conv26', downsample=True)

        for i in range(8):
            input_data = utils.layers.residual_block(input_data, 512, 256, 512,
                                                     is_training=is_training, name='residual{}'.format(i + 11))
        route_2 = input_data
        input_data = utils.layers.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                                is_training=is_training, name='conv43', downsample=True)

        for i in range(4):
            input_data = utils.layers.residual_block(input_data, 1024, 512, 1024,
                                                     is_training=is_training, name='residual{}'.format(i + 19))
        return route_1, route_2, input_data


if __name__ == '__main__':
    input_tensor = tf.placeholder(tf.float32, shape=[None, 320, 320, 3], name='inputs')
    r1, r2, output_tensor = darknet53(input_tensor, is_training=True)

    print(r1)
    print(r2)
    print(output_tensor)
