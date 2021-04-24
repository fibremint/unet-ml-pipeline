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

parser.add_argument('--dataset-cache', dest='is_use_dataset_cache', action='store_true')
parser.add_argument('--no-dataset-cache', dest='is_use_dataset_cache', action='store_false')
parser.set_defaults(is_use_dataset_cache=False)
parser.add_argument('--dataset-cache-path', type=str, default='.', help='path where dataset cache is stored')

parser.add_argument('--image_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--num-class', type=int, default=2, help='# of classes')

parser.add_argument('--data-path', type=str, default='/opt/data', help='data base path')
parser.add_argument('--slide-patch-dir', type=str, default='slide-patch', help='directory '
                                                                               'where slide patches are placed')

parser.add_argument('--checkpoint-metadata-filename', type=str, default='test-checkpoint-metadata.json')

parser.add_argument('--pixel-anno-ignore', type=int, default=44)
parser.add_argument('--pixel-anno-positive', type=int, default=255)
parser.add_argument('--pixel-anno-negative', type=int, default=155)

parser.add_argument('--patch-zoom-range', type=float, default=0.2)

parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--target-checkpoints-path', type=str, default='')

parser.add_argument('--prediction-threshold', type=float, default=0.5, help='')

opt = parser.parse_args()
