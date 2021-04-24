import json
import os

from opts import opt


def main():
    with open(os.path.join(opt.data_path, opt.model_deploy_config_filename)) as f:
        deploy_config_dict = json.load(f)

    print(f'simple model deployer')
    print()
    print(f'deployed checkpoint info:')
    print(f'updated: {deploy_config_dict["update_time"]}')
    print(f'path: {deploy_config_dict["checkpoint_path"]}')
    print(f'metrics: {deploy_config_dict["metrics"]}')


if __name__ == '__main__':
    main()
