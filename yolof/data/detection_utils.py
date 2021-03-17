import numpy as np

import detectron2.data.transforms as T
from detectron2.structures import BoxMode

from .augmentation_impl import (
    YOLOFJitterCrop,
    YOLOFResize,
    YOLOFRandomDistortion,
    RandomFlip,
    YOLOFRandomShift
)


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    is_normal_aug = not cfg.INPUT.RESIZE.ENABLED
    if is_normal_aug:
        augmentation = build_normal_augmentation(cfg, is_train)
    else:
        augmentation = build_yolo_augmentation(cfg, is_train)
    if is_train:
        augmentation.append(
            YOLOFRandomShift(max_shifts=cfg.INPUT.SHIFT.SHIFT_PIXELS))
    return augmentation


def build_normal_augmentation(cfg, is_train):
    """
    Train Augmentations:
        - ResizeShortestEdge
        - RandomFlip (not for test)
    Test:
        - ResizeShortestEdge
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )
    return augmentation


def build_yolo_augmentation(cfg, is_train):
    """
    Train Augmentations:
        - YOLOFJitterCrop
        - YOLOFResize
        - YOLOFRandomDistortion
        - RandomFlip
    Test:
        - YOLOFResize
    """
    augmentation = []
    if is_train:
        if cfg.INPUT.JITTER_CROP.ENABLED:
            augmentation.append(YOLOFJitterCrop(
                cfg.INPUT.JITTER_CROP.JITTER_RATIO))
        augmentation.append(
            YOLOFResize(shape=cfg.INPUT.RESIZE.SHAPE,
                        scale_jitter=cfg.INPUT.RESIZE.SCALE_JITTER)
        )
        if cfg.INPUT.DISTORTION.ENABLED:
            augmentation.append(
                YOLOFRandomDistortion(
                    hue=cfg.INPUT.DISTORTION.HUE,
                    saturation=cfg.INPUT.DISTORTION.SATURATION,
                    exposure=cfg.INPUT.DISTORTION.EXPOSURE
                )
            )
        if cfg.INPUT.RANDOM_FLIP != "none":
            # The difference between `T.RandomFlip` and `RandomFlip` is that
            # we register a new method `apply_meta_infos` in `RandomFlip`
            augmentation.append(
                RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )
    else:
        augmentation.append(
            YOLOFResize(shape=cfg.INPUT.RESIZE.TEST_SHAPE,
                        scale_jitter=None)
        )
    return augmentation


def transform_instance_annotations(
        annotation, transforms, image_size, *, add_meta_infos=False
):
    """
    Apply transforms to box and meta_infos annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        add_meta_infos (bool): Whether to apply meta_infos.

    Returns:
        dict:
            the same input dict with fields "bbox", "meta_infos"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(
        annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    # add meta_infos
    if add_meta_infos:
        meta_infos = dict()
        meta_infos = transforms.apply_meta_infos(meta_infos)
        annotation["meta_infos"] = meta_infos
    return annotation
