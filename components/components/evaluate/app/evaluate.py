from collections import defaultdict
import json
import os

import tensorflow as tf

from data import load_dataset
from model import UNet
from opts import opt


tf.random.set_seed(opt.seed)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


def evaluate_model(target_checkpoints_path,
                   eval_required_checkpoint_paths: list):

    @tf.function
    def _test_step(inputs, labels, weights):
        predictions = model(inputs, training=False)
        pred_loss = _loss_fn(labels=labels, label_weights=weights, predictions=predictions)

        predictions = tf.nn.softmax(predictions, axis=-1)
        predictions = predictions[..., 1]
        predictions = tf.where(predictions > opt.prediction_threshold, 1, 0)

        test_loss.update_state(pred_loss)
        test_mean_iou.update_state(labels, predictions)

    def _loss_fn(labels, label_weights, predictions):
        cross_entropy_loss_pixel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predictions)
        cross_entropy_loss_pixel = tf.multiply(cross_entropy_loss_pixel, label_weights)
        cross_entropy_loss = tf.reduce_sum(cross_entropy_loss_pixel) / (tf.reduce_sum(label_weights) + 0.00001)

        # if opt.weight_decay > 0:
        #     cross_entropy_loss = cross_entropy_loss + opt.weight_decay * tf.add_n(
        #         [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()
        #          if 'batch_normalization' not in v.name])

        return cross_entropy_loss

    test_dataset, test_batch_per_epoch_num = load_dataset(data_type='test')

    model = UNet().create_model(img_shape=[opt.image_size, opt.image_size, 3], num_class=opt.num_class)

    test_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_mean_iou = tf.keras.metrics.MeanIoU(num_classes=opt.num_class)

    eval_checkpoint_dict = defaultdict(dict)

    checkpoints_num = len(eval_required_checkpoint_paths)
    for idx_checkpoint, curr_checkpoint in enumerate(eval_required_checkpoint_paths):
        curr_checkpoint_name = curr_checkpoint.split(os.path.sep)[-1]
        curr_checkpoint_name = os.path.splitext(curr_checkpoint_name)[0]

        model.load_weights(curr_checkpoint)

        for idx_step, (images, labels, weights) in enumerate(test_dataset.take(test_batch_per_epoch_num)):
            _test_step(inputs=images, labels=labels, weights=weights)

        print(f'evaluate ({idx_checkpoint+1}/{checkpoints_num}) '
              f'loss: {format(test_loss.result(), ".7f")} '
              f'mean-iou: {format(test_mean_iou.result(), ".7f")}', flush=True)

        eval_checkpoint_dict[curr_checkpoint_name]['loss'] = float(test_loss.result().numpy())
        eval_checkpoint_dict[curr_checkpoint_name]['mean_iou'] = float(test_mean_iou.result().numpy())

        with open(os.path.join(target_checkpoints_path, opt.checkpoint_metadata_filename), 'w') as f:
            f.write(json.dumps(eval_checkpoint_dict))

        test_loss.reset_states()
        test_mean_iou.reset_states()
