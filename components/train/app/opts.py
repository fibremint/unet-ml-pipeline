'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:20:01
 * @modify date 2017-05-25 02:20:01
 * @desc [description]
'''

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=6, help='# of epochs')
parser.add_argument('--batch-size', type=int, default=16, help='input batch size')

parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lr-decay', dest='is_use_lr_decay', action='store_true')
parser.add_argument('--no-lr-decay', dest='is_use_lr_decay', action='store_false')
parser.set_defaults(is_use_lr_decay=False)
parser.add_argument('--lr-decay-rate', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--lr-decay-epoch', type=int, default=1, help='how many epoch to decay learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
parser.add_argument('--drop-rate', type=float, default=0.0, help='drop rate of unet')

parser.add_argument('--train-iter-epoch-ratio', type=float, default=1.0, help='# of ratio of total images as an epoch')

parser.add_argument('--dataset-cache', dest='is_use_dataset_cache', action='store_true')
parser.add_argument('--no-dataset-cache', dest='is_use_dataset_cache', action='store_false')
parser.set_defaults(is_use_dataset_cache=False)
parser.add_argument('--dataset-cache-path', type=str, default='.', help='path where dataset cache is stored')

parser.add_argument('--image_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--num-class', type=int, default=2, help='# of classes')

parser.add_argument('--data-path', type=str, default='/opt/data', help='data base path')
parser.add_argument('--slide-patch-dir', type=str, default='slide-patch', help='directory '
                                                                               'where slide patches are placed')
# parser.add_argument('--log-path', type=str, default='./log', help='where tensorflow summary is saved')
parser.add_argument('--checkpoint-dir', type=str, default='model-checkpoint', help='where checkpoint saved')
parser.add_argument('--save-checkpoint', dest='is_save_checkpoint', action='store_true')
parser.add_argument('--no-save-checkpoint', dest='is_save_checkpoint', action='store_false')
parser.set_defaults(is_save_checkpoint=False)
# parser.add_argument('--checkpoint-metadata-root-filename', type=str, default='train-checkpoint.json')
parser.add_argument('--checkpoint-metadata-filename', type=str, default='train-checkpoint-metadata.json')

parser.add_argument('--pixel-anno-ignore', type=int, default=44)
parser.add_argument('--pixel-anno-positive', type=int, default=255)
parser.add_argument('--pixel-anno-negative', type=int, default=155)

parser.add_argument('--patch-zoom-range', type=float, default=0.2)

parser.add_argument('--seed', type=int, default=42)

opt = parser.parse_args()
