from typing import Any, Dict, Union, Tuple

import cv2
import numpy as np

from detectron2.data.transforms import (
    Transform,
    HFlipTransform,
    VFlipTransform,
    ResizeTransform
)

__all__ = [
    "Transform",
    "YOLOFJitterCropTransform",
    "YOLOFDistortTransform",
    "HFlipTransform",
    "VFlipTransform",
    "ResizeTransform",
    "YOLOFShiftTransform"
]


@Transform.register_type("meta_infos")
def apply_meta_infos(transform: Transform, meta_infos: Dict) -> Dict:
    """
    Add a new field to save meta information.
    """
    return meta_infos


class YOLOFJitterCropTransform(Transform):
    """
    JitterCrop data augmentation used in YOLOv4.

    Steps:
        - get random offset of four boundaries
        - get target crop size
        - get target crop image
        - get cropped coordinates

    Args:
        pleft (int): left offset.
        pright (int): right offset.
        ptop (int): top offset.
        pbot (int): bottom offset.
        output_size (tuple(int)): output size (w, h).
    """

    def __init__(self,
                 pleft: int,
                 pright: int,
                 ptop: int,
                 pbot: int,
                 output_size: Tuple[Union[int, Any], Union[int, Any]]):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the cropped image(s).
        """
        oh, ow = img.shape[:2]

        swidth, sheight = self.output_size

        # x1,y1,x2,y2
        src_rect = [
            self.pleft, self.ptop, swidth + self.pleft, sheight + self.ptop
        ]
        img_rect = [0, 0, ow, oh]
        # rect intersection
        new_src_rect = [
            max(src_rect[0], img_rect[0]),
            max(src_rect[1], img_rect[1]),
            min(src_rect[2], img_rect[2]),
            min(src_rect[3], img_rect[3])
        ]
        dst_rect = [
            max(0, -self.pleft),
            max(0, -self.ptop),
            max(0, -self.pleft) + new_src_rect[2] - new_src_rect[0],
            max(0, -self.ptop) + new_src_rect[3] - new_src_rect[1]
        ]

        # crop the image
        cropped = np.zeros([sheight, swidth, 3], dtype=img.dtype)
        cropped[:, :, :] = np.mean(img, axis=(0, 1))
        cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
            img[new_src_rect[1]:new_src_rect[3],
            new_src_rect[0]:new_src_rect[2]]
        return cropped

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Crop the coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.

        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H 1 - y)`.
        """
        coords_offset = np.array([self.pleft, self.ptop], dtype=np.float32)
        coords = coords - coords_offset
        swidth, sheight = self.output_size
        coords[..., 0] = np.clip(coords[..., 0], 0, swidth - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, sheight - 1)
        return coords

    def apply_meta_infos(self, meta_infos: Dict) -> Dict:
        """
        Add a new apply function for `JitterCropTransform`. As we need to
        use the jitter offsets in Mosaic. (See `YOLOFDtasetMapper`)

        Args:
            meta_infos (Dict): Jitter crop meta_infos.

        Returns:
            meta_infos (Dict): Updated meta infos.
        """
        meta_infos["jitter_pad_left"] = self.pleft
        meta_infos["jitter_pad_right"] = self.pright
        meta_infos["jitter_pad_top"] = self.ptop
        meta_infos["jitter_pad_bot"] = self.pbot
        return meta_infos


class YOLOFDistortTransform(Transform):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, hue, saturation, exposure):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        dhue = np.random.uniform(low=-self.hue, high=self.hue)
        dsat = self._rand_scale(self.saturation)
        dexp = self._rand_scale(self.exposure)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img = np.asarray(img, dtype=np.float32) / 255.
        img[:, :, 1] *= dsat
        img[:, :, 2] *= dexp
        H = img[:, :, 0] + dhue * 179 / 255.

        if dhue > 0:
            H[H > 1.0] -= 1.0
        else:
            H[H < 0.0] += 1.0

        img[:, :, 0] = H
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = np.asarray(img, dtype=np.float32)

        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def _rand_scale(self, upper_bound):
        """
        Calculate random scaling factor.

        Args:
            upper_bound (float): range of the random scale.
        Returns:
            random scaling factor (float) whose range is
            from 1 / s to s .
        """
        scale = np.random.uniform(low=1, high=upper_bound)
        if np.random.rand() > 0.5:
            return scale
        return 1 / scale


@HFlipTransform.register_type("meta_infos")
def apply_meta_infos(hflip_transform: Transform, meta_infos: Dict) -> Dict:
    pleft = meta_infos["jitter_pad_left"]
    pright = meta_infos["jitter_pad_right"]
    pleft, pright = pright, pleft
    meta_infos["jitter_pad_left"] = pleft
    meta_infos["jitter_pad_right"] = pright
    return meta_infos


@VFlipTransform.register_type("meta_infos")
def apply_meta_infos(vflip_transform: Transform, meta_infos: Dict) -> Dict:
    ptop = meta_infos["jitter_pad_top"]
    pbot = meta_infos["jitter_pad_bot"]
    ptop, pbot = pbot, ptop
    meta_infos["jitter_pad_top"] = ptop
    meta_infos["jitter_pad_bot"] = pbot
    return meta_infos


@ResizeTransform.register_type("meta_infos")
def apply_meta_infos(resize_transform: Transform, meta_infos: Dict) -> Dict:
    scale_w = resize_transform.new_w * 1.0 / resize_transform.w
    scale_h = resize_transform.new_h * 1.0 / resize_transform.h
    meta_infos["jitter_pad_left"] *= scale_w
    meta_infos["jitter_pad_right"] *= scale_w
    meta_infos["jitter_pad_top"] *= scale_h
    meta_infos["jitter_pad_bot"] *= scale_h
    return meta_infos


class YOLOFShiftTransform(Transform):
    """
    Shift the image with random pixels.
    As the `YOLOFShiftTransform` is the last transform, it do not need to
    deal with `meta_infos`.
    """

    def __init__(self, shift_x: int, shift_y: int):
        """
        Args:
            shift_x (int): the shift pixel for x axis.
            shift_y (int): the shift pixel for y axis.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Shift the image(s).

        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: shifted image(s).
        """
        new_img = np.zeros_like(img)
        if self.shift_x < 0:
            new_x = 0
            orig_x = - self.shift_x
        else:
            new_x = self.shift_x
            orig_x = 0
        if self.shift_y < 0:
            new_y = 0
            orig_y = - self.shift_y
        else:
            new_y = self.shift_y
            orig_y = 0

        if len(img.shape) <= 3:
            img_h, img_w = img.shape[:2]
            new_h = img_h - np.abs(self.shift_y)
            new_w = img_w - np.abs(self.shift_x)
            new_img[new_y:new_y + new_h, new_x:new_x + new_w] \
                = img[orig_y:orig_y + new_h, orig_x:orig_x + new_w]
            return new_img
        else:
            img_h, img_w = img.shape[1:3]
            new_h = img_h - np.abs(self.shift_y)
            new_w = img_w - np.abs(self.shift_x)
            new_img[..., new_y:new_y + new_h, new_x:new_x + new_w, :] \
                = img[..., orig_y:orig_y + new_h, orig_x:orig_x + new_w, :]
            return new_img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply shift transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2.
                Each row is (x, y).

        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] += self.shift_x
        coords[:, 1] += self.shift_y
        return coords

    def apply_meta_infos(self, meta_infos: Dict) -> Dict:
        """
        Apply shift transform on meta_infos.
        """
        meta_infos["jitter_pad_left"] = max(
            0, meta_infos["jitter_pad_left"] - self.shift_x)
        meta_infos["jitter_pad_right"] = max(
            0, meta_infos["jitter_pad_right"] + self.shift_x)
        meta_infos["jitter_pad_top"] = max(
            0, meta_infos["jitter_pad_top"] - self.shift_y)
        meta_infos["jitter_pad_bot"] = max(
            0, meta_infos["jitter_pad_bot"] + self.shift_y)
        return meta_infos
