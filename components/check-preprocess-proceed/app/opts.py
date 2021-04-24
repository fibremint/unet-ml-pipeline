import argparse


parser = argparse.ArgumentParser(description='Check whether slide pre-processing is required')

parser.add_argument('--data-path', type=str, default='/opt/data',
                    help='data directory, contains annotation, pre-process checkpoint and slide')
parser.add_argument('--slide-dir', type=str, default='slide',
                    help='slide directory')
parser.add_argument('--preprocess-checkpoint-filename', type=str, default='preprocess-checkpoint.json',
                    help='slide filenames that pre-processed successfully')

opt = parser.parse_args()
