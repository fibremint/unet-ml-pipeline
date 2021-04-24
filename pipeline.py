import os

from kfp import dsl, compiler

import pipeline_ops as po


@dsl.pipeline(name='ct-pipeline')
def nwd_pipeline():
    batch_size = 64
    gpu_num = 1
    seed = 42

    eval_prediction_threshold = 0.5

    # tf_memory_limit = '7G'

    check_preprocess_proceed_task = po.check_preprocess_proceed_op()
    is_preprocess_required = check_preprocess_proceed_task.outputs['is_preprocess_required']

    with dsl.Condition(is_preprocess_required == 'True', name='preprocess-required'):
        preprocess_required_slide_paths = check_preprocess_proceed_task.outputs['preprocess_required_slide_paths']

        data_preprocess_task = po.data_preprocess_op(preprocess_required_slide_paths)
        data_preprocess_task.after(check_preprocess_proceed_task)

        train_task = po.train_op(batch_size=batch_size,
                                 seed=seed)

        train_task.set_gpu_limit(gpu_num)
        # train_task.set_memory_limit(tf_memory_limit)
        # train_task.set_cpu_limit('1000m')

        train_task.after(data_preprocess_task)
        target_checkpoints_path = train_task.outputs['target_checkpoints_path']

        eval_task = po.evaluate_op(target_checkpoints_path=target_checkpoints_path,
                                   prediction_threshold=eval_prediction_threshold,
                                   seed=seed)
        eval_task.set_gpu_limit(gpu_num)
        eval_task.after(train_task)

        target_checkpoints_path = eval_task.outputs['target_checkpoints_path']
        checkpoint_metrics_filename = eval_task.outputs['checkpoint_metrics_filename']

        check_deployable_task = po.check_deployable_op(target_checkpoints_path=target_checkpoints_path,
                                                       checkpoint_metrics_filename=checkpoint_metrics_filename)

        check_deployable_task.after(eval_task)

        deployable_checkpoint_info = check_deployable_task.outputs['deployable_checkpoint_info']
        is_deployable_checkpoint_exist = check_deployable_task.outputs['is_deployable_checkpoint_exist']

        with dsl.Condition(is_deployable_checkpoint_exist == 'True', name='exist'):
            update_deploy_config_task = po.update_deploy_config_op(
                deployable_checkpoint_info=deployable_checkpoint_info)

            update_deploy_config_task.after(check_deployable_task)

            deploy_task = po.deploy_op()
            deploy_task.after(update_deploy_config_task)

        with dsl.Condition(is_deployable_checkpoint_exist == 'False', name='not-exist'):
            po.print_op('deployed checkpoint: previous')

    with dsl.Condition(is_preprocess_required == 'False', name='preprocess-not-required'):
        print_task = po.print_op('all of the slides are pre-processed')


if __name__ == '__main__':
    #ref: https://towardsdatascience.com/build-your-data-pipeline-on-kubernetes-using-kubeflow-pipelines-sdk-and-argo-eef69a80237c
    workflow_dict = compiler.Compiler()._create_workflow(nwd_pipeline)
    # ref: https://github.com/argoproj/argo-workflows/issues/4534
    # Failed to establish pod watch: timed out waiting for the condition
    workflow_dict['metadata']['namespace'] = 'argo'
    workflow_dict['spec']['serviceAccountName'] = 'default'

    if not os.path.exists('./output'):
        os.makedirs('./output')

    compiler.Compiler._write_workflow(workflow_dict, './output/pipeline.yml')
