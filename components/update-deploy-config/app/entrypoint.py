import datetime
import glob
import json
import os

from model import UNet
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

    model_serve_path = os.path.join(opt.data_path, opt.seldon_model_path, 'nwd')
    if not os.path.exists(model_serve_path):
        os.makedirs(model_serve_path)

    saved_model_paths = glob.glob(model_serve_path + '/*')
    if len(saved_model_paths) > 0:
        latest_model_version = max([int(os.path.relpath(g, model_serve_path))
                                    for g in glob.glob(model_serve_path + '/*')])
    else:
        latest_model_version = 0

    model_save_path = os.path.join(model_serve_path, str(latest_model_version + 1))
    print(f'msp: {model_save_path}')
    load_weight_path = os.path.join(opt.data_path, deployable_checkpoint_dict['checkpoint_path'])
    print(f'lwp: {load_weight_path}')

    model = UNet().create_model(img_shape=[256, 256, 3], num_class=2, rate=.0)
    model.load_weights(load_weight_path)
    model.save(model_save_path)


if __name__ == '__main__':
    main()
