import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data-path', type=str, default='/opt/data',
                    help='data directory, contains annotation, pre-process checkpoint and slide')
# parser.add_argument('--checkpoint-dir', type=str, default='model-checkpoint', help='where checkpoint saved')
parser.add_argument('--target-checkpoints-path', type=str, default='')
parser.add_argument('--checkpoint-metrics-filename', type=str, default='')
parser.add_argument('--model-deploy-config-filename', type=str, default='model-deploy-config.json')

opt = parser.parse_args()
