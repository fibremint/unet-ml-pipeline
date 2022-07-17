import glob
import json
import os

from opts import opt


def main(args):
    slide_path = os.path.join(args.data_path, args.slide_dir)
    preprocess_checkpoint_path = os.path.join(args.data_path, args.preprocess_checkpoint_filename)

    if not os.path.exists(preprocess_checkpoint_path):
        with open(preprocess_checkpoint_path, 'w+') as f:
            f.write(json.dumps([]))

    preprocessed_checkpoint = json.load(open(preprocess_checkpoint_path))

    # ref: https://stackoverflow.com/questions/44994604/python-glob-os-relative-path-making-filenames-into-a-list
    preprocess_required_slide_paths = [os.path.relpath(slide_file_path, slide_path)
                                       for slide_file_path in glob.glob(slide_path + '/*/*')
                                       if slide_file_path.split(os.path.sep)[-1] not in preprocessed_checkpoint]

    is_preprocess_required = len(preprocess_required_slide_paths) != 0
    with open('/tmp/is-preprocess-required.txt', 'w') as f:
        f.write(str(is_preprocess_required))

    with open('/tmp/preprocess-required-slide-paths.json', 'w') as f:
        f.write(json.dumps(preprocess_required_slide_paths))


if __name__ == '__main__':
    main(opt)
