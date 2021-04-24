import tensorflow as tf


# ref: https://stackoverflow.com/questions/38376478/changing-the-scale-of-a-tensor-in-tensorflow
def normalize(tensor):
    return tf.divide(
        tf.subtract(
            tensor,
            tf.reduce_min(tensor)
        ),
        tf.subtract(
            tf.reduce_max(tensor),
            tf.reduce_min(tensor)
        )
    )
