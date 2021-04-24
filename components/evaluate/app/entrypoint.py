import glob
import os
import sys

from evaluate import evaluate_model
from opts import opt


def main():
    current_checkpoints_path = os.path.join(opt.data_path, opt.target_checkpoints_path)
    eval_required_checkpoint_paths = glob.glob(current_checkpoints_path + '*/*.h5')

    evaluate_model(target_checkpoints_path=current_checkpoints_path,
                   eval_required_checkpoint_paths=eval_required_checkpoint_paths)

    with open('/tmp/target-checkpoints-path.txt', 'w') as f:
        f.write(opt.target_checkpoints_path)

    with open('/tmp/checkpoint-metrics-filename.txt', 'w') as f:
        f.write(opt.checkpoint_metadata_filename)


if __name__ == '__main__':
    main()
