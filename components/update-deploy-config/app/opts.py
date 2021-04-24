import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data-path', type=str, default='/opt/data',
                    help='data directory, contains annotation, pre-process checkpoint and slide')
parser.add_argument('--model-deploy-config-filename', type=str, default='model-deploy-config.json')

parser.add_argument('--deployable-checkpoint-info', type=str, default='')

opt = parser.parse_args()
