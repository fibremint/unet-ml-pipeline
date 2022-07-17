"""
slide patch: a pair of image and ground-truth from slide
"""
import os
import pathlib
import glob
import tensorflow as tf

from opts import opt
from util import image as image_util


AUTOTUNE = tf.data.experimental.AUTOTUNE

ground_truth_relative_path = ''
if os.name == 'nt':
    ground_truth_relative_path += os.sep

ground_truth_relative_path += os.sep + 'ground-truth'


def _load_slide_patch(image_path, image_size):
    ground_truth_path = tf.strings.regex_replace(image_path, "[\\\/](image)", ground_truth_relative_path)

    image = image_util.load(image_path, size_h=image_size, size_w=image_size, channels=3)
    ground_truth_image = image_util.load(ground_truth_path, size_h=image_size, size_w=image_size, channels=1)

    return image, ground_truth_image


def _parse_ground_truth_image(ground_truth_image,
                              positive_value=opt.pixel_anno_positive,
                              ignore_value=opt.pixel_anno_ignore):

    ground_truth = tf.cast(ground_truth_image, dtype=tf.int32)[..., 0]

    label = tf.where(ground_truth == positive_value, 1, 0)
    weight = tf.where(ground_truth == ignore_value, 0.5, 1)

    return label, weight


# wraps with tf.function to crop on the same region in the each of images
# ref: https://www.tensorflow.org/api_docs/python/tf/random/set_seed?version=nightly
@tf.function
def _slide_patch_random_zoom(image, ground_truth, image_original_size, image_zoom_range: float, random_crop_seed=42):
    image_crop_coefficient = tf.random.uniform([1], minval=1.0 - image_zoom_range, maxval=1)
    image_crop_size = tf.multiply(tf.cast(image_original_size, dtype=tf.float32), image_crop_coefficient)
    image_crop_size = tf.cast(image_crop_size, dtype=tf.int32)

    original_image_crop_size = tf.concat([image_crop_size, image_crop_size, [3]], axis=0)
    label_image_crop_size = tf.concat([image_crop_size, image_crop_size, [1]], axis=0)

    image = tf.image.random_crop(image, size=original_image_crop_size, seed=random_crop_seed)
    image = tf.image.resize(image,
                            size=(image_original_size, image_original_size),
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    ground_truth = tf.image.random_crop(ground_truth, size=label_image_crop_size, seed=random_crop_seed)
    ground_truth = tf.image.resize(ground_truth,
                                   size=(image_original_size, image_original_size),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, ground_truth


def _load_train_data(image_path,
                     image_size,
                     image_zoom_range,
                     is_flip_horizontal=False,
                     flip_probability=0.5):

    image, ground_truth_image = _load_slide_patch(image_path, image_size=image_size)

    image, ground_truth_image = _slide_patch_random_zoom(image, ground_truth_image,
                                                         image_original_size=image_size,
                                                         image_zoom_range=image_zoom_range)

    if is_flip_horizontal and tf.random.uniform([1]) < flip_probability:
        image = tf.image.flip_left_right(image)
        ground_truth_image = tf.image.flip_left_right(ground_truth_image)

    image = image_util.normalize(image)
    label, weight = _parse_ground_truth_image(ground_truth_image)

    return image, label, weight


def _load_test_data(image_path, image_size):
    image, ground_truth_image = _load_slide_patch(image_path, image_size=image_size)

    image = image_util.normalize(image)
    label, weight = _parse_ground_truth_image(ground_truth_image)

    return image, label, weight


def _prepare_train_dataset(dataset: tf.data.Dataset, batch_size, cache_path='', shuffle_buffer_size=1000):
    if cache_path != '':
        cache_filename = 'dataset_train.tfcache'
        dataset = dataset.cache(os.path.join(opt.data_path, cache_path, cache_filename))
        # dataset = dataset.cache(''.join([cache_path, '/', cache_filename]))

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # repeat forever
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)

    # `prefetch` lets the dataset fetch batches in the background
    # while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def _prepare_test_dataset(dataset: tf.data.Dataset, batch_size, cache_path=''):
    if cache_path != '':
        cache_filename = 'dataset_test.tfcache'
        dataset = dataset.cache(os.path.join(opt.data_path, cache_path, cache_filename))
        # dataset = dataset.cache(''.join([cache_path, '/', cache_filename]))

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)

    return dataset


def load_dataset(data_type=''):
    if data_type == '':
        raise Exception('data type is not specified to load')

    avail_data_type = ('train', 'test')
    if data_type not in avail_data_type:
        raise Exception(f'data type {data_type} is not available to load')

    cache_path = ''
    if opt.is_use_dataset_cache:
        cache_path = opt.dataset_cache_path

    data_root_path = pathlib.Path(opt.data_path)
    image_dir = 'image/*/*.png'

    if data_type == 'train':
        train_image_path_str = str(data_root_path / str('slide-patch/train/'+image_dir))
        # train_image_path_str = str(data_root_path / str('model-train/'+image_dir))

        train_data_len = len(glob.glob(train_image_path_str))
        tf.print(f'[INFO] model-train data: #{train_data_len}')
        train_batch_per_epoch_num = int(train_data_len / opt.batch_size * opt.train_iter_epoch_ratio)

        train_image_path_list = tf.data.Dataset.list_files(train_image_path_str)

        # map with additional params
        # ref: https://stackoverflow.com/questions/46263963/how-to-map-a-function-with-additional-parameter-using-the-new-dataset-api-in-tf1
        train_dataset = train_image_path_list.map(lambda image_path: _load_train_data(image_path,
                                                                                      image_size=opt.image_size,
                                                                                      image_zoom_range=opt.patch_zoom_range,
                                                                                      is_flip_horizontal=True),
                                                  num_parallel_calls=AUTOTUNE)

        train_dataset = _prepare_train_dataset(train_dataset,
                                               batch_size=opt.batch_size,
                                               cache_path=cache_path,
                                               shuffle_buffer_size=1000)

        return train_dataset, train_batch_per_epoch_num

    elif data_type == 'test':
        test_image_path_str = str(data_root_path / str('slide-patch/test/'+image_dir))
        test_data_len = len(glob.glob(test_image_path_str))
        tf.print(f'[INFO] test data: #{test_data_len}')
        test_batch_per_epoch_num = int(test_data_len / opt.batch_size)

        test_image_path_list = tf.data.Dataset.list_files(test_image_path_str)

        test_dataset = test_image_path_list.map(lambda image_path: _load_test_data(image_path,
                                                                                   image_size=opt.image_size))

        test_dataset = _prepare_test_dataset(test_dataset,
                                             cache_path=cache_path,
                                             batch_size=opt.batch_size)

        return test_dataset, test_batch_per_epoch_num
    #
    # return train_dataset, test_dataset, train_batch_per_epoch_num, test_batch_per_epoch_num
