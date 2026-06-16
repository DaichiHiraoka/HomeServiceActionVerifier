from __future__ import annotations

import numpy as np

from home_service_action_verifier.masking import (
    background_blur_with_roi,
    face_like_top_mask,
    grid_image,
    lower_body_only,
    no_mask,
    object_area_only,
)
from home_service_action_verifier.schemas import FrameInfo, ROI


def _image() -> np.ndarray:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :, 0] = 80
    image[50:90, 60:100] = (20, 200, 40)
    return image


def test_mask_functions_keep_image_size() -> None:
    image = _image()
    roi = ROI(x1=50, y1=45, x2=110, y2=95)
    masked_images = [
        no_mask(image),
        face_like_top_mask(image),
        background_blur_with_roi(image, roi),
        lower_body_only(image),
        object_area_only(image, roi),
    ]
    assert all(masked.shape == image.shape for masked in masked_images)


def test_grid_image_generates_image() -> None:
    frames = [_image(), _image(), _image()]
    infos = [FrameInfo(frame_index=index, timestamp=index / 10) for index in range(3)]
    grid = grid_image(frames, infos)
    assert grid.shape[0] > 0
    assert grid.shape[1] > 0
    assert grid.shape[2] == 3
