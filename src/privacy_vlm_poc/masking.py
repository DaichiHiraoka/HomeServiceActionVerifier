"""Privacy-preserving image masking and grid composition helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from privacy_vlm_poc.schemas import FrameInfo, MaskMethod, ROI


def _coerce_roi(roi: ROI | tuple[int, int, int, int] | None, width: int, height: int) -> ROI:
    if roi is None:
        default = ROI(
            x1=int(width * 0.25),
            y1=int(height * 0.45),
            x2=int(width * 0.75),
            y2=int(height * 0.95),
        )
        return default.clipped(width, height)
    if isinstance(roi, ROI):
        return roi.clipped(width, height)
    return ROI(x1=roi[0], y1=roi[1], x2=roi[2], y2=roi[3]).clipped(width, height)


def no_mask(image: np.ndarray) -> np.ndarray:
    return image.copy()


def face_like_top_mask(image: np.ndarray) -> np.ndarray:
    """Mask the upper center region without detecting or identifying faces."""

    masked = image.copy()
    height, width = masked.shape[:2]
    mask_width = int(width * 0.45)
    mask_height = int(height * 0.32)
    x1 = max(0, (width - mask_width) // 2)
    x2 = min(width, x1 + mask_width)
    y1 = 0
    y2 = min(height, mask_height)
    masked[y1:y2, x1:x2] = 0
    return masked


def background_blur_with_roi(
    image: np.ndarray,
    roi: ROI | tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    height, width = image.shape[:2]
    region = _coerce_roi(roi, width, height)
    blur_kernel = max(15, (min(width, height) // 18) | 1)
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    blurred[region.y1 : region.y2, region.x1 : region.x2] = image[region.y1 : region.y2, region.x1 : region.x2]
    return blurred


def lower_body_only(image: np.ndarray) -> np.ndarray:
    masked = image.copy()
    height = masked.shape[0]
    masked[: height // 2, :] = 0
    return masked


def object_area_only(
    image: np.ndarray,
    roi: ROI | tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    height, width = image.shape[:2]
    region = _coerce_roi(roi, width, height)
    masked = np.zeros_like(image)
    masked[region.y1 : region.y2, region.x1 : region.x2] = image[region.y1 : region.y2, region.x1 : region.x2]
    return masked


def apply_mask(
    image: np.ndarray,
    mask_method: str | MaskMethod,
    roi: ROI | tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    method = MaskMethod(mask_method)
    if method == MaskMethod.NONE:
        return no_mask(image)
    if method == MaskMethod.FACE_LIKE_TOP_MASK:
        return face_like_top_mask(image)
    if method == MaskMethod.BACKGROUND_BLUR_WITH_ROI:
        return background_blur_with_roi(image, roi)
    if method == MaskMethod.FULL_FRAME_BLUR_EXCEPT_ROI:
        return background_blur_with_roi(image, roi)
    if method == MaskMethod.LOWER_BODY_ONLY:
        return lower_body_only(image)
    if method == MaskMethod.OBJECT_AREA_ONLY:
        return object_area_only(image, roi)
    if method == MaskMethod.HAND_OBJECT_ROI:
        return object_area_only(image, roi)
    if method == MaskMethod.TOKEN_ONLY:
        return no_mask(image)
    msg = f"Unsupported mask method: {mask_method}"
    raise ValueError(msg)


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        msg = f"Failed to read image: {path}"
        raise ValueError(msg)
    return image


def grid_image(
    frames: Sequence[np.ndarray],
    frame_infos: Sequence[FrameInfo] | None = None,
    columns: int | None = None,
) -> np.ndarray:
    """Create one VLM-friendly grid image with compact frame labels."""

    if not frames:
        raise ValueError("grid_image requires at least one frame")

    first_height, first_width = frames[0].shape[:2]
    columns = columns or int(math.ceil(math.sqrt(len(frames))))
    rows = int(math.ceil(len(frames) / columns))
    grid = np.zeros((rows * first_height, columns * first_width, 3), dtype=np.uint8)

    for index, frame in enumerate(frames):
        row = index // columns
        col = index % columns
        if frame.shape[:2] != (first_height, first_width):
            frame = cv2.resize(frame, (first_width, first_height), interpolation=cv2.INTER_AREA)
        cell = frame.copy()
        if frame_infos and index < len(frame_infos):
            info = frame_infos[index]
            label = f"#{index + 1} frame {info.frame_index}  {info.timestamp:.2f}s"
        else:
            label = f"#{index + 1} frame {index}"
        cv2.rectangle(cell, (0, 0), (min(first_width, 280), 34), (0, 0, 0), thickness=-1)
        cv2.putText(
            cell,
            label,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y1 = row * first_height
        x1 = col * first_width
        grid[y1 : y1 + first_height, x1 : x1 + first_width] = cell

    return grid
