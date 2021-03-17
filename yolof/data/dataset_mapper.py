from collections import deque
import copy
import logging
from typing import Optional, List, Union

import numpy as np
import torch

from detectron2.config import configurable, CfgNode
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.structures import BoxMode

from .detection_utils import build_augmentation, transform_instance_annotations


class YOLOFDtasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by YOLOF.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Add a queue for saving previous image infos in mosaic transformation
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(self,
                 is_train: bool,
                 *,
                 augmentations: List[Union[T.Augmentation, T.Transform]],
                 image_format: str,
                 mosaic_trans: Optional[CfgNode],
                 use_instance_mask: bool = False,
                 use_keypoint: bool = False,
                 instance_mask_format: str = "polygon",
                 recompute_boxes: bool = False,
                 add_meta_infos: bool = False):
        """
        Args:
            augmentations: a list of augmentations or deterministic
                transforms to apply
            image_format: an image format supported by
                :func:`detection_utils.read_image`.
            mosaic_trans: a CfgNode for Mosaic transformation.
            use_instance_mask: whether to process instance segmentation
                annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process
                instance segmentation masks into this format.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask
                annotations.
            add_meta_infos: whether to add `meta_infos` field
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.use_instance_mask = use_instance_mask
        self.instance_mask_format = instance_mask_format
        self.use_keypoint = use_keypoint
        self.recompute_boxes = recompute_boxes
        self.add_meta_infos = add_meta_infos
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

        self.mosaic_trans = mosaic_trans
        if self.mosaic_trans.ENABLED:
            self.mosaic_pool = deque(
                maxlen=self.mosaic_trans.POOL_CAPACITY)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        # use a local `build_augmentation` instead
        augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0,
                        T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        elif cfg.INPUT.JITTER_CROP.ENABLED and is_train:
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "mosaic_trans": cfg.INPUT.MOSAIC,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            "add_meta_infos": cfg.INPUT.JITTER_CROP.ENABLED
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset
                format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(
            dataset_dict)  # it will be modified by code below
        # add image info to mosaic pool
        mosaic_flag = 0
        mosaic_samples = None
        if self.mosaic_trans.ENABLED and self.is_train:
            if len(self.mosaic_pool) > self.mosaic_trans.NUM_IMAGES:
                mosaic_flag = np.random.randint(2)
                # sample images in the mosaic_pool
                if mosaic_flag == 1:
                    mosaic_samples = np.random.choice(
                        self.mosaic_pool,
                        self.mosaic_trans.NUM_IMAGES - 1)
            self.mosaic_pool.append(copy.deepcopy(dataset_dict))

        # for current image
        image, annos = self._load_image_with_annos(dataset_dict)

        if self.is_train and mosaic_flag == 1 and mosaic_samples is not None:
            min_offset = self.mosaic_trans.MIN_OFFSET
            mosaic_width = self.mosaic_trans.MOSAIC_WIDTH
            mosaic_height = self.mosaic_trans.MOSAIC_HEIGHT
            cut_x = np.random.randint(int(mosaic_width * min_offset),
                                      int(mosaic_width * (1 - min_offset)))
            cut_y = np.random.randint(int(mosaic_height * min_offset),
                                      int(mosaic_height * (1 - min_offset)))
            # init the out image and the out annotations
            out_image = np.zeros(
                [mosaic_height, mosaic_width, 3], dtype=image.dtype)
            out_annos = []
            # mosaic transform
            for m_idx in range(self.mosaic_trans.NUM_IMAGES):
                # re-load the image and annotations for the sampled images
                # replace the current image and annos with the new image's
                if m_idx != 0:
                    dataset_dict = copy.deepcopy(mosaic_samples[m_idx - 1])
                    image, annos = self._load_image_with_annos(dataset_dict)

                image_size = image.shape[:2]  # h, w
                # as all meta_infos are the same, we just get the first one
                meta_infos = annos[0].pop("meta_infos")
                pleft = meta_infos.get('jitter_pad_left', 0)
                pright = meta_infos.get('jitter_pad_right', 0)
                ptop = meta_infos.get('jitter_pad_top', 0)
                pbot = meta_infos.get('jitter_pad_bot', 0)
                # get shifts
                left_shift = min(cut_x, max(0, -int(pleft)))
                top_shift = min(cut_y, max(0, -int(ptop)))
                right_shift = min(image_size[1] - cut_x, max(0, -int(pright)))
                bot_shift = min(image_size[0] - cut_y, max(0, -int(pbot)))
                out_image, cur_annos = self._blend_moasic(
                    cut_x,
                    cut_y,
                    out_image,
                    image,
                    copy.deepcopy(annos),
                    (mosaic_height, mosaic_width),
                    m_idx,
                    (left_shift, top_shift, right_shift, bot_shift)
                )
                out_annos.extend(cur_annos)
            # replace image and annotation with out_image and out_annotation
            image, annos = out_image, out_annos

        if annos is not None:
            image_shape = image.shape[:2]  # h, w
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box
            # may no longer tightly bound the object. As an example, imagine
            # a triangle object [(0,0), (2,0), (0,2)] cropped by a box [(1,
            # 0),(2,2)] (XYXY format). The tight bounding box of the cropped
            # triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # Pytorch's dataloader is efficient on torch.Tensor due to
        # shared-memory,
        # but not efficient on large generic data structures due to the use
        # of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        return dataset_dict

    def _load_image_with_annos(self, dataset_dict):
        """
        Load the image and annotations given a dataset_dict.
        """
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"],
                                 format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return image, None

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other
            # types of data
            # apply meta_infos for mosaic transformation
            annos = [
                transform_instance_annotations(
                    obj, transforms, image_shape,
                    add_meta_infos=self.add_meta_infos
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
        else:
            annos = None
        return image, annos

    def _apply_boxes(self,
                     annotations,
                     left_shift,
                     top_shift,
                     cut_width,
                     cut_height,
                     cut_start_x,
                     cut_start_y):
        """
        Modify the boxes' coordinates according to shifts and cut_starts.
        """
        for annotation in annotations:
            bboxes = BoxMode.convert(annotation["bbox"],
                                     annotation["bbox_mode"],
                                     BoxMode.XYXY_ABS)
            bboxes = np.asarray(bboxes)
            bboxes[0::2] -= left_shift
            bboxes[1::2] -= top_shift

            bboxes[0::2] = np.clip(bboxes[0::2], 0, cut_width)
            bboxes[1::2] = np.clip(bboxes[1::2], 0, cut_height)
            bboxes[0::2] += cut_start_x
            bboxes[1::2] += cut_start_y
            annotation["bbox"] = bboxes
            annotation["bbox_mode"] = BoxMode.XYXY_ABS
        return annotations

    def _blend_moasic(self,
                      cut_x,
                      cut_y,
                      target_img,
                      img,
                      annos,
                      img_size,
                      blend_index,
                      four_shifts):
        """
        Blend the images and annotations in Mosaic transform.
        """
        h, w = img_size
        img_h, img_w = img.shape[:2]
        left_shift = min(four_shifts[0], img_w - cut_x)
        top_shift = min(four_shifts[1], img_h - cut_y)
        right_shift = min(four_shifts[2], img_w - (w - cut_x))
        bot_shift = min(four_shifts[3], img_h - (h - cut_y))

        if blend_index == 0:
            annos = self._apply_boxes(
                annos, left_shift, top_shift, cut_x, cut_y, 0, 0
            )
            target_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y,
                                         left_shift:left_shift + cut_x]
        if blend_index == 1:
            annos = self._apply_boxes(
                annos, img_w + cut_x - w - right_shift,
                top_shift, w - cut_x, cut_y, cut_x, 0
            )
            target_img[:cut_y, cut_x:] = \
                img[top_shift:top_shift + cut_y,
                img_w + cut_x - w - right_shift:img_w - right_shift]
        if blend_index == 2:
            annos = self._apply_boxes(
                annos, left_shift, img_h + cut_y - h - bot_shift,
                cut_x, h - cut_y, 0, cut_y
            )
            target_img[cut_y:, :cut_x] = \
                img[img_h + cut_y - h - bot_shift:img_h - bot_shift,
                left_shift:left_shift + cut_x]
        if blend_index == 3:
            annos = self._apply_boxes(annos, img_w + cut_x - w - right_shift,
                                      img_h + cut_y - h - bot_shift,
                                      w - cut_x, h - cut_y, cut_x, cut_y)
            target_img[cut_y:, cut_x:] = \
                img[img_h + cut_y - h - bot_shift:img_h - bot_shift,
                img_w + cut_x - w - right_shift:img_w - right_shift]
        return target_img, annos
