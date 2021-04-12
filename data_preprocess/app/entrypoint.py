"""
Pre-process the data to generate the image and mask patch.

this script checks if the slides that had to be processed are exist by checking the checkpoint file. if that slides
are exist, slide pre-processing would be started.
"""
import os
import sys
import uuid
import json
import argparse
import glob

from skimage import io

from annotation import load_annotation
from slide import preprocess_slide


if __name__ == "__main__":
    '''
    
    args:
        slide: whole slide image
        slide_annotation: corresponding annotation to slide
        
    output data path structure:
        
    '''
    parser = argparse.ArgumentParser(description='Bladder slide annotation loading and visulization')
    parser.add_argument('--data-dir', type=str, default='/opt/data',
                        help='data directory, contains annotation, pre-process checkpoint and slide')
    parser.add_argument('--crop_size',   type=int, default=1024)
    parser.add_argument('--save_size',   type=int, default=256)
    parser.add_argument('--slide_level', type=int, default=0)

    args = parser.parse_args()

    annotation_dir = os.path.join(args.data_dir, 'region_annotation')
    if not os.path.exists(annotation_dir):
        raise Exception(f'Not found an annotation directory: {annotation_dir}')

    slide_dir = os.path.join(args.data_dir, 'slide')
    preprocess_checkpoint_path = os.path.join(args.data_dir, 'preprocess_checkpoint.json')
    if not os.path.exists(preprocess_checkpoint_path):
        with open(preprocess_checkpoint_path, 'w+') as f:
            f.write(json.dumps([]))

    preprocess_metadata_dir = os.path.join(args.data_dir, 'preprocess_metadata')
    if not os.path.exists(preprocess_metadata_dir):
        os.makedirs(preprocess_metadata_dir)

    slide_patch_dir = os.path.join(args.data_dir, 'slide_patch')
    if not os.path.exists(slide_patch_dir):
        os.makedirs(slide_patch_dir)

    preprocessed_checkpoint = json.load(open(preprocess_checkpoint_path))
    to_process_slide_paths = [slide_path for slide_path in glob.glob(slide_dir+'/*/*')
                              if slide_path.split(os.path.sep)[-1] not in preprocessed_checkpoint]

    if not to_process_slide_paths:
        print(f'all of the slides are pre-processed')
        sys.exit(0)

    _, annotation_labels, _ = next(os.walk(annotation_dir))
    for curr_slide_path in to_process_slide_paths:
        data_type, slide_filename = curr_slide_path.split(os.path.sep)[-2:]
        slide_id = slide_filename.split('.')[0]

        preprocessed_metadata = list()
        preprocessed_metadata_path = os.path.join(preprocess_metadata_dir, f'{slide_filename}.json')
        with open(preprocessed_metadata_path, 'w') as f:
            f.write(json.dumps(preprocessed_metadata))

        for curr_annotation_label in annotation_labels:
            annotation_path = os.path.join(annotation_dir, curr_annotation_label, slide_id, "annotations.json")
            if not os.path.exists(annotation_path):
                print(f'Not found an annotation of {curr_annotation_label} for slide {slide_filename}, '
                      f'pre-processing has been skipped')
                continue

            annotation_dict = load_annotation(annotation_path)

            image_save_relative_dir = '/'.join([data_type, 'image', curr_annotation_label])
            image_save_base_path = os.path.join(slide_patch_dir, image_save_relative_dir)
            if not os.path.exists(image_save_base_path):
                os.makedirs(image_save_base_path)

            mask_save_relative_dir = '/'.join([data_type, 'ground_truth', curr_annotation_label])
            mask_save_base_path = os.path.join(slide_patch_dir, mask_save_relative_dir)
            if not os.path.exists(mask_save_base_path):
                os.makedirs(mask_save_base_path)

            pre_processed_data_iter = preprocess_slide(slide_path=curr_slide_path,
                                                       annotation_dict=annotation_dict,
                                                       annotation_label=curr_annotation_label,
                                                       slide_level=args.slide_level,
                                                       crop_size=args.crop_size,
                                                       save_size=args.save_size)

            for image, ground_truth in pre_processed_data_iter:
                filename = str(uuid.uuid4())[:8]+".png"
                image_save_path = os.path.join(image_save_base_path, filename)
                mask_save_path = os.path.join(mask_save_base_path, filename)

                io.imsave(image_save_path, image)
                image_save_relative_path = '/'.join([image_save_relative_dir, filename])
                preprocessed_metadata.append(image_save_relative_path)

                io.imsave(mask_save_path, ground_truth, check_contrast=False)
                mask_save_relative_path = '/'.join([mask_save_relative_dir, filename])
                preprocessed_metadata.append(mask_save_relative_path)

                with open(preprocessed_metadata_path, 'w') as f:
                    f.write(json.dumps(preprocessed_metadata))

        print(f'pre-processing finished: {slide_filename}')
        preprocessed_checkpoint.append(slide_filename)

        with open(preprocess_checkpoint_path, 'w') as f:
            json.dump(preprocessed_checkpoint, f)
