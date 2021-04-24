import datetime
import json
import os

from opts import opt


def main():
    model_deploy_config_path = os.path.join(opt.data_path, opt.model_deploy_config_filename)
    with open(model_deploy_config_path, 'r') as f:
        model_deploy_config = json.load(f)

    deployable_checkpoint_dict = json.loads(opt.deployable_checkpoint_info)

    model_deploy_config['checkpoint_path'] = deployable_checkpoint_dict['checkpoint_path']
    model_deploy_config['metrics'] = deployable_checkpoint_dict['metrics']
    model_deploy_config['update_time'] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    with open(model_deploy_config_path, 'w') as f:
        f.write(json.dumps(model_deploy_config))


if __name__ == '__main__':
    main()
