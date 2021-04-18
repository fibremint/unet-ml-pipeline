import datetime
import os

from opts import opt
from train import train_model


def main():
    checkpoints_path = ''
    current_checkpoint_rel_path = ''

    if opt.is_save_checkpoint:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        current_checkpoint_rel_path = os.path.join(opt.checkpoint_dir, current_time)
        checkpoints_path = os.path.join(opt.data_path, current_checkpoint_rel_path)

        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

    train_model(checkpoints_path=checkpoints_path)

    if opt.is_save_checkpoint:
        with open('/tmp/target-checkpoints-path.txt', 'w') as f:
            f.write(current_checkpoint_rel_path)

        with open('/tmp/is-checkpoint-eval-required.txt', 'w') as f:
            f.write(str(True))

    else:
        with open('/tmp/target-checkpoints-path.txt', 'w') as f:
            f.write('')

        with open('/tmp/is-checkpoint-eval-required.txt', 'w') as f:
            f.write(str(False))


if __name__ == '__main__':
    main()
