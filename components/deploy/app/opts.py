import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data-path', type=str, default='/opt/data')
parser.add_argument('--model-deploy-config-filename', type=str, default='model-deploy-config.json')

opt = parser.parse_args()
