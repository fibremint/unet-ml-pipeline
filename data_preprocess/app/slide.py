import json
import os

import geopandas as gpd
import numpy as np
import openslide
from geocube.api.core import make_geocube
from pycontour import poly_transform
from shapely.geometry import Point, Polygon, mapping, box
from skimage import transform

NUM_TRY_GENERATE_PATCH = 10

PIXEL_ANNO_IGN = 44
PIXEL_ANNO_POS = 255
PIXEL_ANNO_NEG = 155

annotation_label_dict = {'positive': PIXEL_ANNO_POS, 'negative': PIXEL_ANNO_NEG}


def _generate_ground_truth(w, h, crop_size, annotation_polygon: Polygon, annotation_label):
    """

    :param slide:
    :return: image, mask
    # """

    patch_mask_polygon = Polygon([(w, h),
                                  (w + crop_size, h),
                                  (w + crop_size, h + crop_size),
                                  (w, h + crop_size)])
    patch_mask_polygon = gpd.GeoSeries(patch_mask_polygon)
    annotation_polygon = gpd.GeoSeries(annotation_polygon)

    # Get the intersection of the `patch mask and an annotation
    #
    # 'patch_mask' would fed into GeoDataFrame as dataset.
    gdf_mask = gpd.GeoDataFrame({'geometry': patch_mask_polygon, 'patch_mask': annotation_label_dict[annotation_label]})

    gdf_curr_annotation = gpd.GeoDataFrame({'geometry': annotation_polygon})
    gdf_mask_curr_anno_diff = gpd.overlay(gdf_mask, gdf_curr_annotation, how='intersection')

    if not gdf_mask_curr_anno_diff.empty:
        # 'geom' work as boundary box
        mask_curr_anno_intersection_rasterized = \
            make_geocube(vector_data=gdf_mask_curr_anno_diff,
                         resolution=(1., 1.),
                         geom=json.dumps(mapping(box(w, h, w+crop_size, h+crop_size))),
                         fill=PIXEL_ANNO_IGN)

        # TODO: refactor a transformation of geocube data to numpy array
        intersection_data = mask_curr_anno_intersection_rasterized.to_dict()
        intersection_data = intersection_data['data_vars']['patch_mask']['data']
        patch_ground_truth = np.array(intersection_data)

        return patch_ground_truth

    return np.full((crop_size, crop_size), annotation_label_dict[annotation_label]).astype(np.float)


def preprocess_slide(slide_path,
                     annotation_dict,
                     annotation_label,
                     slide_level,
                     crop_size,
                     save_size):
    """
    generates the image and mask patch from slide, and save them.

    :param crop_size:
    :param save_size:
    :param slide_path:
    :param annotation_dict:
    :param annotation_label:
    :param slide_level:
    :return:
    """

    slide = openslide.OpenSlide(slide_path)
    slide_name = os.path.basename(slide_path)

    if slide_level < 0 or slide_level >= slide.level_count:
        raise Exception(f'level {slide_level} is not available in the {slide_name}')

    if annotation_label not in annotation_label_dict:
        raise Exception(f"a value of an annotation '{annotation_label}' is not set")

    for curr_annotation_region in annotation_dict:
        annotation_coords = \
            (annotation_dict[curr_annotation_region] / slide.level_downsamples[slide_level]).astype(np.int32)
        annotation_coords = np.transpose(np.array(annotation_coords))
        # swap width and height
        annotation_coords[[0, 1]] = annotation_coords[[1, 0]]
        min_h, max_h = np.min(annotation_coords[0, :]), np.max(annotation_coords[0, :])
        min_w, max_w = np.min(annotation_coords[1, :]), np.max(annotation_coords[1, :])

        annotation_polygon = poly_transform.np_arr_to_poly(np.asarray(annotation_coords))

        patch_generate_try_cnt = 0
        while patch_generate_try_cnt < NUM_TRY_GENERATE_PATCH:
            rand_h = np.random.randint(min_h, max_h)
            rand_w = np.random.randint(min_w, max_w)
            is_height_exceeds = rand_h + crop_size >= slide.level_dimensions[slide_level][1]
            is_width_exceeds = rand_w + crop_size >= slide.level_dimensions[slide_level][0]
            if is_width_exceeds or is_height_exceeds:
                continue

            patch_center_coord_h = int(rand_h + crop_size / 2)
            patch_center_coord_w = int(rand_w + crop_size / 2)
            patch_center_point = Point(patch_center_coord_w, patch_center_coord_h)
            if not patch_center_point.within(annotation_polygon):
                patch_generate_try_cnt += 1
                continue

            curr_patch = slide.read_region((rand_w, rand_h), slide_level, (crop_size, crop_size))
            curr_patch = np.asarray(curr_patch)[:, :, :3]
            curr_patch_ground_truth = _generate_ground_truth(rand_w, rand_h,
                                                             crop_size=crop_size,
                                                             annotation_polygon=annotation_polygon,
                                                             annotation_label=annotation_label)

            curr_patch = transform.resize(curr_patch, (save_size, save_size))

            # order=0: Nearest-neighbor interpolation
            curr_patch_ground_truth = transform.resize(curr_patch_ground_truth,
                                                       (save_size, save_size),
                                                       order=0,
                                                       anti_aliasing=False)

            curr_patch = (curr_patch * 255).astype(np.uint8)
            curr_patch_ground_truth = curr_patch_ground_truth.astype(np.uint8)

            yield curr_patch, curr_patch_ground_truth

            patch_generate_try_cnt += 1
