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


def train_model(checkpoints_path):

    @tf.function
    def _train_step(inputs, labels, weights):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            pred_loss = _loss_fn(labels=labels, label_weights=weights, predictions=predictions)

        gradients = tape.gradient(pred_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        predictions = tf.argmax(predictions, axis=-1)

        train_loss.update_state(pred_loss)
        train_mean_iou.update_state(labels, predictions)

    def _loss_fn(labels, label_weights, predictions):
        cross_entropy_loss_pixel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=predictions)
        cross_entropy_loss_pixel = tf.multiply(cross_entropy_loss_pixel, label_weights)
        cross_entropy_loss = tf.reduce_sum(cross_entropy_loss_pixel) / (tf.reduce_sum(label_weights) + 0.00001)

        if opt.weight_decay > 0:
            cross_entropy_loss = cross_entropy_loss + opt.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()
                 if 'batch_normalization' not in v.name])

        return cross_entropy_loss

    # train_data_set, test_data_set, train_batch_per_epoch_num, test_batch_per_epoch_num \
    #     = load_dataset()
    train_dataset, train_batch_per_epoch_num = load_dataset(data_type='train')

    model = UNet().create_model(img_shape=[opt.image_size, opt.image_size, 3], num_class=opt.num_class,
                                rate=opt.drop_rate)

    optimizer = None
    if opt.is_use_lr_decay:
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            opt.learning_rate,
            decay_steps=train_batch_per_epoch_num // opt.lr_decay_epoch + 1,
            decay_rate=opt.lr_decay_rate,
            staircase=True)

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate_schedule)
    else:
        optimizer = tf.optimizers.Adam(learning_rate=opt.learning_rate)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_mean_iou = tf.keras.metrics.MeanIoU(num_classes=opt.num_class)

    train_checkpoint_dict = defaultdict(dict)
    # train_checkpoint_filename = 'train-checkpoint'
    # with open(os.path.join(checkpoints_path, train_checkpoint_filename), 'w') as f:
    #     f.write(json.dumps(train_checkpoint_dict))

    for idx_epoch in range(opt.epoch):
        for idx_step, (images, labels, weights) in enumerate(train_dataset.take(train_batch_per_epoch_num)):
            _train_step(images, labels, weights)

        learning_rate = optimizer._decayed_lr(tf.float32)

        print(f'epoch ({idx_epoch+1}/{opt.epoch}) '
              f'learning-rate: {format(learning_rate, ".7f")} '
              f'loss: {format(train_loss.result(), ".7f")} '
              f'mean-iou: {format(train_mean_iou.result(), ".7f")}', flush=True)

        if opt.is_save_checkpoint:
            checkpoint_name = f'checkpoint-epoch-{idx_epoch+1}'

            train_checkpoint_dict[checkpoint_name]['loss'] = float(train_loss.result().numpy())
            train_checkpoint_dict[checkpoint_name]['mean_iou'] = float(train_mean_iou.result().numpy())

            with open(os.path.join(checkpoints_path, opt.checkpoint_metadata_filename), 'w') as f:
                f.write(json.dumps(train_checkpoint_dict))

            model.save_weights(os.path.join(checkpoints_path, f'{checkpoint_name}.h5'))

        train_loss.reset_states()
        train_mean_iou.reset_states()
