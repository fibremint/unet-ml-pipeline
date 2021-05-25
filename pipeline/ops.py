from kfp import dsl, components

REPO_ADDRESS = '<repo_base>/<gcp_project_id>'

DATA_PATH = '/opt/data/nwd-data'
SELDON_MODEL_PATH = '/opt/data/seldon-model'
VOL_PVC_NAME = 'nfs-pv-claim'


@dsl.pipeline(
    name='Check Preprocess Proceed',
    description='check slide pre-processing would be required'
)
def check_preprocess_proceed_op():
    return dsl.ContainerOp(
        name='check-preprocess-proceed',
        image=f'{REPO_ADDRESS}/nwd-check-preprocess-proceed',
        arguments=['--data-path', DATA_PATH],
        file_outputs={'is_preprocess_required': '/tmp/is-preprocess-required.txt',
                      'preprocess_required_slide_paths': '/tmp/preprocess-required-slide-paths.json'},
        pvolumes={'/opt/data': dsl.PipelineVolume(pvc=VOL_PVC_NAME)}
    )


@dsl.pipeline(
    name='Data Preprocess',
    description='Pre-processing on a slide'
)
def data_preprocess_op(preprocess_required_slide_paths: str):
    return dsl.ContainerOp(
        name='data-preprocess',
        image=f'{REPO_ADDRESS}/nwd-data-preprocess',
        arguments=['--data-path', DATA_PATH,
                   '--preprocess-required-slide-paths', preprocess_required_slide_paths],
        pvolumes={'/opt/data': dsl.PipelineVolume(pvc=VOL_PVC_NAME)}
    )


@dsl.pipeline(
    name='Train',
    description='train the model'
)
def train_op(batch_size: int, seed: int):
    return dsl.ContainerOp(
        name='train',
        image=f'{REPO_ADDRESS}/nwd-train',
        arguments=['--data-path', DATA_PATH,
                   '--batch-size', batch_size,
                   '--save-checkpoint',
                   '--seed', seed,
                   '--lr-decay'],
        file_outputs={'target_checkpoints_path': '/tmp/target-checkpoints-path.txt',
                      'is_checkpoint_eval_required': '/tmp/is-checkpoint-eval-required.txt'},
        pvolumes={'/opt/data': dsl.PipelineVolume(pvc=VOL_PVC_NAME)}
    )


@dsl.pipeline(
    name='Evaluate',
    description='evaluate the generated checkpoints'
)
def evaluate_op(target_checkpoints_path: str, prediction_threshold, seed: int):
    return dsl.ContainerOp(
        name='evaluate',
        image=f'{REPO_ADDRESS}/nwd-evaluate',
        arguments=['--data-path', DATA_PATH,
                   '--target-checkpoints-path', target_checkpoints_path,
                   '--prediction-threshold', prediction_threshold,
                   '--seed', seed],
        file_outputs={'target_checkpoints_path': '/tmp/target-checkpoints-path.txt',
                      'checkpoint_metrics_filename': '/tmp/checkpoint-metrics-filename.txt'},
        pvolumes={'/opt/data': dsl.PipelineVolume(pvc=VOL_PVC_NAME)}
    )


@dsl.pipeline(
    name='Check deployable',
    description='check checkpoint is exist which is more optimal than deployed one'
)
def check_deployable_op(target_checkpoints_path: str, checkpoint_metrics_filename: str):
    return dsl.ContainerOp(
        name='check-deployable',
        image=f'{REPO_ADDRESS}/nwd-check-deployable',
        arguments=['--data-path', DATA_PATH,
                   '--target-checkpoints-path', target_checkpoints_path,
                   '--checkpoint-metrics-filename', checkpoint_metrics_filename],
        file_outputs={'is_deployable_checkpoint_exist': '/tmp/is-deployable-checkpoint-exist.txt',
                      'deployable_checkpoint_info': '/tmp/deployable-checkpoint-info.json'},
        pvolumes={'/opt/data': dsl.PipelineVolume(pvc=VOL_PVC_NAME)}
    )


@dsl.pipeline(
    name='Update deploy config',
    description='update deploy config'
)
def update_deploy_config_op(deployable_checkpoint_info: str):
    return dsl.ContainerOp(
        name='update-deploy-config',
        image=f'{REPO_ADDRESS}/nwd-update-deploy-config',
        arguments=['--data-path', DATA_PATH,
                   '--seldon-model-path', SELDON_MODEL_PATH,
                   '--deployable-checkpoint-info', deployable_checkpoint_info],
        pvolumes={'/opt/data': dsl.PipelineVolume(pvc=VOL_PVC_NAME)}
    )


@dsl.pipeline(
    name='Simple model deployer',
    description='shows model deploy config'
)
def deploy_op():
    return dsl.ContainerOp(
        name='deploy',
        image=f'{REPO_ADDRESS}/nwd-deploy',
        arguments=['--data-path', DATA_PATH],
        pvolumes={'/opt/data': dsl.PipelineVolume(pvc=VOL_PVC_NAME)}
    )


def _print_fn(msg) -> None:
    print(msg)


print_op = components.func_to_container_op(_print_fn)
