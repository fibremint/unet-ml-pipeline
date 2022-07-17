import argparse

parser = argparse.ArgumentParser(description='Generate the patches of cropped image patch and '
                                             'corresponding annotation image')

parser.add_argument('--preprocess-required-slide-paths', type=str, default='[]')

parser.add_argument('--data-path', type=str, default='/opt/data',
                    help='data directory, contains annotation, pre-process checkpoint and slide')
parser.add_argument('--slide-dir', type=str, default='slide',
                    help='slide directory')
parser.add_argument('--slide-patch-dir', type=str, default='slide-patch',
                    help='directory for save patch and annotation images from pre-processing of the slide')
parser.add_argument('--annotation-dir', type=str, default='region-annotation')
parser.add_argument('--preprocess-metadata-dir', type=str, default='preprocess-metadata')
parser.add_argument('--preprocess-checkpoint-filename', type=str, default='preprocess-checkpoint.json',
                    help='slide filenames that pre-processed successfully')
parser.add_argument('--annotation-filename', type=str, default='annotations.json')

parser.add_argument('--crop-size', type=int, default=1024)
parser.add_argument('--save-size', type=int, default=256)
parser.add_argument('--slide-level', type=int, default=0)

parser.add_argument('--patch-gen-try-num', type=int, default=10)

parser.add_argument('--pixel-anno-ignore', type=int, default=44)
parser.add_argument('--pixel-anno-positive', type=int, default=255)
parser.add_argument('--pixel-anno-negative', type=int, default=155)

opt = parser.parse_args()
