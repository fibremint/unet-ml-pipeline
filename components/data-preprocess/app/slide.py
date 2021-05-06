import logging
import json
import os
import uuid

import geopandas as gpd
import numpy as np
import openslide
from geocube.api.core import make_geocube
from pycontour import poly_transform
from shapely.geometry import Point, Polygon, mapping, box
from skimage import transform, io

from annotation import load_annotation
from opts import opt


annotation_label_dict = {'positive': opt.pixel_anno_positive,
                         'negative': opt.pixel_anno_negative}


def _generate_ground_truth(w, h, crop_size, annotation_polygon: Polygon, pixel_annotation_value):
    """

    :param w:
    :param h:
    :param crop_size:
    :param annotation_polygon:
    :param pixel_annotation_value:
    :return:
    """

    patch_mask_polygon = Polygon([(w, h),
                                  (w + crop_size, h),
                                  (w + crop_size, h + crop_size),
                                  (w, h + crop_size)])
    patch_mask_polygon = gpd.GeoSeries(patch_mask_polygon)
    annotation_polygon = gpd.GeoSeries(annotation_polygon)

    # Get the intersection of the `patch mask and an annotation
    #
    # 'patch_mask' would fed into GeoDataFrame as dataset.
    gdf_mask = gpd.GeoDataFrame({'geometry': patch_mask_polygon, 'patch_mask': pixel_annotation_value})

    gdf_curr_annotation = gpd.GeoDataFrame({'geometry': annotation_polygon})
    gdf_mask_curr_anno_diff = gpd.overlay(gdf_mask, gdf_curr_annotation, how='intersection')

    if not gdf_mask_curr_anno_diff.empty:
        # 'geom' work as boundary box
        mask_curr_anno_intersection_rasterized = \
            make_geocube(vector_data=gdf_mask_curr_anno_diff,
                         resolution=(1., 1.),
                         geom=json.dumps(mapping(box(w, h, w+crop_size, h+crop_size))),
                         fill=opt.pixel_anno_ignore)

        # TODO: refactor a transformation of geocube data to numpy array
        intersection_data = mask_curr_anno_intersection_rasterized.to_dict()
        intersection_data = intersection_data['data_vars']['patch_mask']['data']
        patch_ground_truth = np.array(intersection_data)

        return patch_ground_truth

    return np.full((crop_size, crop_size), pixel_annotation_value).astype(np.float)


def _slide_patch_generator(slide_path,
                           annotation_dict,
                           annotation_label):
    """
    generates the image and mask patch from slide, and save them.

    :param slide_path:
    :param annotation_dict:
    :param annotation_label:
    :return:
    """

    slide = openslide.OpenSlide(slide_path)
    slide_name = os.path.basename(slide_path)

    if opt.slide_level < 0 or opt.slide_level >= slide.level_count:
        raise Exception(f'level {opt.slide_level} is not available in the {slide_name}')

    if annotation_label not in annotation_label_dict:
        raise Exception(f"a value of an annotation '{annotation_label}' is not set")

    for curr_annotation_region in annotation_dict:
        annotation_coords = \
            (annotation_dict[curr_annotation_region] / slide.level_downsamples[opt.slide_level]).astype(np.int32)
        annotation_coords = np.transpose(np.array(annotation_coords))
        # swap width and height
        annotation_coords[[0, 1]] = annotation_coords[[1, 0]]
        min_h, max_h = np.min(annotation_coords[0, :]), np.max(annotation_coords[0, :])
        min_w, max_w = np.min(annotation_coords[1, :]), np.max(annotation_coords[1, :])

        try:
            annotation_polygon = poly_transform.np_arr_to_poly(np.asarray(annotation_coords))
        except ValueError as e:
            logging.error(f'Failed to transform coordinates to polygon, this will be skipped,'
                          f' slide_name: {slide_name}, region: {curr_annotation_region}')
            logging.error(f'Exception: {e}')
            continue

        patch_generate_try_cnt = 0
        while patch_generate_try_cnt < opt.patch_gen_try_num:
            rand_h = np.random.randint(min_h, max_h)
            rand_w = np.random.randint(min_w, max_w)
            is_height_exceeds = rand_h + opt.crop_size >= slide.level_dimensions[opt.slide_level][1]
            is_width_exceeds = rand_w + opt.crop_size >= slide.level_dimensions[opt.slide_level][0]
            if is_width_exceeds or is_height_exceeds:
                continue

            patch_center_coord_h = int(rand_h + opt.crop_size / 2)
            patch_center_coord_w = int(rand_w + opt.crop_size / 2)
            patch_center_point = Point(patch_center_coord_w, patch_center_coord_h)
            if not patch_center_point.within(annotation_polygon):
                patch_generate_try_cnt += 1
                continue

            curr_patch = slide.read_region((rand_w, rand_h), opt.slide_level, (opt.crop_size, opt.crop_size))
            curr_patch = np.asarray(curr_patch)[:, :, :3]
            curr_patch_ground_truth = _generate_ground_truth(rand_w, rand_h,
                                                             crop_size=opt.crop_size,
                                                             annotation_polygon=annotation_polygon,
                                                             pixel_annotation_value=annotation_label_dict[
                                                                 annotation_label])

            curr_patch = transform.resize(curr_patch, (opt.save_size, opt.save_size))

            # order=0: Nearest-neighbor interpolation
            curr_patch_ground_truth = transform.resize(curr_patch_ground_truth,
                                                       (opt.save_size, opt.save_size),
                                                       order=0,
                                                       anti_aliasing=False)

            curr_patch = (curr_patch * 255).astype(np.uint8)
            curr_patch_ground_truth = curr_patch_ground_truth.astype(np.uint8)

            yield curr_patch, curr_patch_ground_truth

            patch_generate_try_cnt += 1


def preprocess(slides_base_path,
               target_slide_paths,
               annotations_base_path,
               annotation_labels,
               annotation_filename,
               slide_patches_base_path,
               preprocess_metadata_base_path,
               preprocess_checkpoint_path):
    """

    :param slides_base_path:
    :param target_slide_paths:
    :param annotations_base_path:
    :param annotation_labels:
    :param annotation_filename:
    :param slide_patches_base_path:
    :param preprocess_metadata_base_path:
    :param preprocess_checkpoint_path:
    :return:
    """

    preprocessed_checkpoint = json.load(open(preprocess_checkpoint_path))

    for curr_slide_path in target_slide_paths:
        data_type, slide_filename = curr_slide_path.split(os.path.sep)
        slide_id = slide_filename.split('.')[0]

        preprocessed_metadata = list()
        preprocessed_metadata_path = os.path.join(preprocess_metadata_base_path, f'{slide_filename}.json')
        with open(preprocessed_metadata_path, 'w') as f:
            f.write(json.dumps(preprocessed_metadata))

        for curr_annotation_label in annotation_labels:
            annotation_path = os.path.join(annotations_base_path, curr_annotation_label,
                                           slide_id, annotation_filename)
            if not os.path.exists(annotation_path):
                print(f'Not found an annotation of {curr_annotation_label} for slide {slide_filename}, '
                      f'pre-processing has been skipped')
                continue

            annotation_dict = load_annotation(annotation_path)

            image_save_relative_dir = '/'.join([data_type, 'image', curr_annotation_label])
            image_save_base_path = os.path.join(slide_patches_base_path, image_save_relative_dir)
            if not os.path.exists(image_save_base_path):
                os.makedirs(image_save_base_path)

            mask_save_relative_dir = '/'.join([data_type, 'ground-truth', curr_annotation_label])
            mask_save_base_path = os.path.join(slide_patches_base_path, mask_save_relative_dir)
            if not os.path.exists(mask_save_base_path):
                os.makedirs(mask_save_base_path)

            pre_processed_data_iter = _slide_patch_generator(slide_path=os.path.join(slides_base_path, curr_slide_path),
                                                             annotation_dict=annotation_dict,
                                                             annotation_label=curr_annotation_label)

            for image, ground_truth in pre_processed_data_iter:
                # _save_patch(image, ground_truth)

                patch_filename = str(uuid.uuid4())[:8] + ".png"

                image_save_path = os.path.join(image_save_base_path, patch_filename)
                mask_save_path = os.path.join(mask_save_base_path, patch_filename)

                io.imsave(image_save_path, image)
                image_save_relative_path = '/'.join([image_save_relative_dir, patch_filename])
                preprocessed_metadata.append(image_save_relative_path)

                io.imsave(mask_save_path, ground_truth, check_contrast=False)
                mask_save_relative_path = '/'.join([mask_save_relative_dir, patch_filename])
                preprocessed_metadata.append(mask_save_relative_path)

                with open(preprocessed_metadata_path, 'w') as f:
                    f.write(json.dumps(preprocessed_metadata))

        if len(preprocessed_metadata) != 0:
            print(f'pre-processing finished: {slide_filename}')
        else:
            print(f'None of the patches are generated from slide: {slide_filename}')

        preprocessed_checkpoint.append(slide_filename)
        with open(preprocess_checkpoint_path, 'w') as f:
            json.dump(preprocessed_checkpoint, f)
