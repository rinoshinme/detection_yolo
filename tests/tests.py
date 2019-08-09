import tensorflow as tf


def test_tile():
    output_size = 5
    y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
    x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_val = sess.run(x)
        y_val = sess.run(y)
        print(x_val)
        print(y_val)


if __name__ == '__main__':
    test_tile()
