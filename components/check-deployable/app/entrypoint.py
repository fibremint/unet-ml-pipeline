import json
import os

from opts import opt

METRIC_KEY = 'mean_iou'
METRIC_OPTIMAL_TYPE = 'max'

avail_metric_optimal_type = ('max', 'min')


def _find_optimal_checkpoint(metrics):
    sorted_metrics = sorted(metrics.items(),
                            key=lambda i: i[1][METRIC_KEY],
                            reverse=METRIC_OPTIMAL_TYPE == 'max')

    return sorted_metrics[0]


def _check_deployable(optimal_checkpoint, model_deploy_config):
    optimal_checkpoint_name, optimal_checkpoint_metrics = optimal_checkpoint

    is_deployable_checkpoint_exist = False

    def _write_deployable_checkpoint_info():
        nonlocal is_deployable_checkpoint_exist

        deployable_checkpoint_info = {
            'checkpoint_path': os.path.join(opt.target_checkpoints_path,
                                            optimal_checkpoint_name),
            'metrics': optimal_checkpoint_metrics
        }
        with open('/tmp/deployable-checkpoint-info.json', 'w') as f:
            f.write(json.dumps(deployable_checkpoint_info))

        is_deployable_checkpoint_exist = True

    is_none_of_deployment = model_deploy_config['checkpoint_path'] == ''
    if is_none_of_deployment:
        _write_deployable_checkpoint_info()

    else:
        if METRIC_OPTIMAL_TYPE == 'max':
            is_optimal_than_deployment = optimal_checkpoint_metrics[METRIC_KEY] > \
                                         model_deploy_config['metrics'][METRIC_KEY]

        else:
            is_optimal_than_deployment = optimal_checkpoint_metrics[METRIC_KEY] < \
                                         model_deploy_config['metrics'][METRIC_KEY]

        if is_optimal_than_deployment:
            _write_deployable_checkpoint_info()

        else:
            with open('/tmp/deployable-checkpoint-info.json', 'w') as f:
                f.write(json.dumps({}))

    with open('/tmp/is-deployable-checkpoint-exist.txt', 'w') as f:
        print(f'idce: {is_deployable_checkpoint_exist}')
        f.write(str(is_deployable_checkpoint_exist))


def main():
    if METRIC_OPTIMAL_TYPE not in avail_metric_optimal_type:
        raise Exception(f'not a valid metric optimal type')

    evaluated_metrics_path = os.path.join(opt.data_path,
                                          opt.target_checkpoints_path,
                                          opt.checkpoint_metrics_filename)

    with open(evaluated_metrics_path, 'r') as f:
        evaluated_metrics = json.load(f)

    optimal_checkpoint = _find_optimal_checkpoint(evaluated_metrics)

    model_deploy_config_path = os.path.join(opt.data_path, opt.model_deploy_config_filename)
    if not os.path.exists(model_deploy_config_path):
        blank_metadata = {
            'checkpoint_path': '',
            'metrics': {}
        }

        with open(model_deploy_config_path, 'w') as f:
            f.write(json.dumps(blank_metadata))

    with open(model_deploy_config_path, 'r') as f:
        model_deploy_config = json.load(f)

    _check_deployable(optimal_checkpoint=optimal_checkpoint,
                      model_deploy_config=model_deploy_config)


if __name__ == '__main__':
    main()
