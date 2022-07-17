"""
Pre-process the data to generate the image and mask patch.

this script checks if the slides that had to be processed are exist by checking the checkpoint file. if that slides
are exist, slide pre-processing would be started.
"""
import json
import os
import sys

from slide import preprocess
from opts import opt


def main(args):
    slides_path = os.path.join(args.data_path, args.slide_dir)
    preprocess_checkpoint_path = os.path.join(args.data_path, args.preprocess_checkpoint_filename)

    annotations_path = os.path.join(args.data_path, args.annotation_dir)
    if not os.path.exists(annotations_path):
        raise Exception(f'Not found an annotation path: {annotations_path}')

    preprocess_metadata_path = os.path.join(args.data_path, args.preprocess_metadata_dir)
    if not os.path.exists(preprocess_metadata_path):
        os.makedirs(preprocess_metadata_path)

    slide_patches_path = os.path.join(args.data_path, args.slide_patch_dir)
    if not os.path.exists(slide_patches_path):
        os.makedirs(slide_patches_path)

    preprocess_required_slide_paths = json.loads(args.preprocess_required_slide_paths)

    if len(preprocess_required_slide_paths) == 0:
        print(f'all of the slides are pre-processed')
        sys.exit(0)

    _, annotation_labels, _ = next(os.walk(annotations_path))

    preprocess(slides_base_path=slides_path,
               target_slide_paths=preprocess_required_slide_paths,
               annotations_base_path=annotations_path,
               annotation_labels=annotation_labels,
               annotation_filename=args.annotation_filename,
               slide_patches_base_path=slide_patches_path,
               preprocess_metadata_base_path=preprocess_metadata_path,
               preprocess_checkpoint_path=preprocess_checkpoint_path)


if __name__ == "__main__":
    '''
    
    args:
        slide: whole slide image
        slide_annotation: corresponding annotation to slide
        
    output data path structure:
        
    '''

    main(opt)
