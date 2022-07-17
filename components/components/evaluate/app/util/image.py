import tensorflow as tf

from .tensor import normalize as normalize_tensor


def load(image_path, size_h, size_w, channels=3):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.resize(image, [size_h, size_w])

    return image


def normalize(image):
    image = normalize_tensor(image)
    image = tf.subtract(image, [0.5])
    image = tf.multiply(image, [2.])

    return image
