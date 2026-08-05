"""Dataset and DataLoader utilities for CGFSDS-9 segmentation.

Provides ``SegmentationDataset`` that loads image / ground-truth mask pairs,
samples foreground and background point prompts, generates bounding boxes,
and applies data augmentation for robust training.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import sam2_lora.config as cfg


def _binarize_mask(mask_arr: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Convert a grayscale mask to binary using a threshold.

    Handles masks with intermediate values caused by compression artifacts
    or soft boundaries.  Values > threshold become 1, others become 0.

    Args:
        mask_arr: Grayscale mask array with values in [0, 255].
        threshold: Binarization threshold (default 127).

    Returns:
        Binary mask array with values in {0.0, 1.0}.
    """
    return (mask_arr > threshold).astype(np.float32)


def _sample_foreground_points(
    mask: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, int]:
    """Sample point coordinates from inside the mask (foreground).

    Args:
        mask: Binary mask of shape (H, W) with values in {0, 1}.
        max_points: Maximum number of points to sample.

    Returns:
        Tuple of (coords, num_points) where coords has shape (N, 2) in
        (row, col) format.
    """
    coords = np.argwhere(mask > 0.5)
    num_fg = coords.shape[0]
    if num_fg == 0:
        h, w = mask.shape
        return np.array([[h // 2, w // 2]], dtype=np.int64), 1
    if num_fg > max_points:
        indices = np.random.choice(num_fg, max_points, replace=False)
        coords = coords[indices]
    return coords, coords.shape[0]


def _sample_background_points(
    mask: np.ndarray,
    num_points: int,
    margin: int = 5,
) -> np.ndarray:
    """Sample point coordinates from outside the mask (background).

    Points are sampled from regions where mask == 0, with a small margin
    from the mask boundary to avoid ambiguous edge pixels.

    Args:
        mask: Binary mask of shape (H, W) with values in {0, 1}.
        num_points: Number of background points to sample.
        margin: Erosion margin to avoid sampling right at the boundary.

    Returns:
        Coordinates array of shape (N, 2) in (row, col) format.
    """
    from scipy.ndimage import binary_erosion

    h, w = mask.shape
    # Erode the mask to get a "safe" foreground region.
    if margin > 0 and mask.sum() > 0:
        eroded = binary_erosion(mask > 0.5, iterations=margin)
    else:
        eroded = mask > 0.5

    # Background = not eroded foreground (avoids boundary ambiguity).
    bg_mask = (~eroded).astype(np.float32)
    bg_coords = np.argwhere(bg_mask > 0.5)

    if bg_coords.shape[0] == 0:
        # Fallback: use corners.
        return np.array([
            [0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1],
        ], dtype=np.int64)[:num_points]

    if bg_coords.shape[0] > num_points:
        indices = np.random.choice(
            bg_coords.shape[0], num_points, replace=False,
        )
        bg_coords = bg_coords[indices]

    return bg_coords


def _sample_edge_points(
    mask: np.ndarray,
    num_points: int,
    band_width: int = 10,
) -> Tuple[np.ndarray, int]:
    """Sample points near the mask boundary for boundary-aware training.

    Creates a band around the mask edge and samples points within it.

    Args:
        mask: Binary mask of shape (H, W) with values in {0, 1}.
        num_points: Maximum number of edge points to sample.
        band_width: Width of the boundary band in pixels.

    Returns:
        Tuple of (coords, num_sampled) in (row, col) format.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.int64), 0

    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(mask > 0.5, structure=struct, iterations=band_width)
    eroded = binary_erosion(mask > 0.5, structure=struct, iterations=band_width)
    edge_band = dilated & (~eroded)

    edge_coords = np.argwhere(edge_band)
    if edge_coords.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64), 0

    if edge_coords.shape[0] > num_points:
        indices = np.random.choice(
            edge_coords.shape[0], num_points, replace=False,
        )
        edge_coords = edge_coords[indices]

    return edge_coords, edge_coords.shape[0]


def _mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """Convert a binary mask to a bounding box [y1, x1, y2, x2].

    Args:
        mask: Binary mask of shape (H, W).

    Returns:
        Bounding box as ``[y1, x1, y2, x2]`` in pixel coordinates.
        Returns ``[0, 0, 0, 0]`` if mask is empty.
    """
    coords = np.argwhere(mask > 0.5)
    if coords.shape[0] == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return np.array([y_min, x_min, y_max, x_max], dtype=np.float32)


def _random_augment(
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random augmentations to image and mask pair.

    Augmentations (applied with 50% probability each):
    - Horizontal flip
    - Vertical flip
    - Random 90-degree rotation

    Args:
        image: RGB image array of shape (H, W, 3).
        mask: Binary mask array of shape (H, W).

    Returns:
        Tuple of (augmented_image, augmented_mask).
    """
    # Horizontal flip.
    if random.random() > 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()

    # Vertical flip.
    if random.random() > 0.5:
        image = np.flip(image, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()

    # Random 90-degree rotation (0, 90, 180, 270).
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k=k).copy()
        mask = np.rot90(mask, k=k).copy()

    return image, mask


class SegmentationDataset(Dataset):
    """PyTorch dataset for image / binary-mask pairs with rich prompts.

    For each sample, the following prompt types are generated:
    - **Foreground points**: sampled from inside the mask (label=1).
    - **Background points**: sampled from outside the mask (label=0).
    - **Edge points**: sampled near the mask boundary (label=1).
    - **Bounding box**: derived from the mask extent.

    Data augmentation (random flips and rotations) is applied during
    training to improve generalization.

    Args:
        data_dir: Root of the split dataset.
        subset: One of ``'train'`` or ``'test'``.
        image_size: Spatial size for resizing.
        max_fg_points: Maximum foreground points per image.
        max_bg_points: Maximum background points per image.
        max_edge_points: Maximum edge points per image.
        augment: Whether to apply random augmentations.
        include_bbox: Whether to include bounding box prompts.
    """

    def __init__(
        self,
        data_dir: str,
        subset: str = 'train',
        image_size: int = cfg.IMAGE_SIZE,
        max_fg_points: int = cfg.MAX_FG_POINTS,
        max_bg_points: int = cfg.MAX_BG_POINTS,
        max_edge_points: int = cfg.MAX_EDGE_POINTS,
        augment: bool = True,
        include_bbox: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.max_fg_points = max_fg_points
        self.max_bg_points = max_bg_points
        self.max_edge_points = max_edge_points
        self.augment = augment and (subset == 'train')
        self.include_bbox = include_bbox
        self.samples: List[Tuple[str, str]] = []

        root = Path(data_dir) / subset
        if not root.is_dir():
            raise FileNotFoundError(f'Subset directory not found: {root}')

        for cat_dir in sorted(root.iterdir()):
            img_dir = cat_dir / 'Images'
            gt_dir = cat_dir / 'GT'
            if not img_dir.is_dir() or not gt_dir.is_dir():
                continue
            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                gt_file = gt_dir / (img_file.stem + '.png')
                if gt_file.is_file():
                    self.samples.append((str(img_file), str(gt_file)))

        if not self.samples:
            raise RuntimeError(
                f'No samples found in {root}. '
                f'Expected: {root / "*/Images/"} and {root / "*/GT/"}.'
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[index]

        image = np.array(Image.open(img_path).convert('RGB'))
        mask_raw = np.array(Image.open(mask_path).convert('L'))
        mask = _binarize_mask(mask_raw, threshold=127)

        # Resize.
        image_pil = Image.fromarray(image).resize(
            (self.image_size, self.image_size), Image.BILINEAR,
        )
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST,
        )
        image = np.array(image_pil)
        mask = _binarize_mask(np.array(mask_pil), threshold=127)

        # Augmentation (training only).
        if self.augment:
            image, mask = _random_augment(image, mask)

        # Image tensor: (3, H, W) in [0, 1].
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        # Mask tensor: (1, H, W) in {0, 1}.
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # --- Point prompts ---
        all_pts = []
        all_labels = []

        # Foreground points (label=1).
        fg_coords, num_fg = _sample_foreground_points(mask, self.max_fg_points)
        fg_xy = fg_coords[:, [1, 0]].astype(np.float32)  # (row,col) -> (x,y)
        all_pts.append(fg_xy)
        all_labels.append(np.ones(num_fg, dtype=np.int64))

        # Background points (label=0).
        if self.max_bg_points > 0:
            bg_coords = _sample_background_points(
                mask, self.max_bg_points, margin=5,
            )
            bg_xy = bg_coords[:, [1, 0]].astype(np.float32)
            all_pts.append(bg_xy)
            all_labels.append(np.zeros(bg_coords.shape[0], dtype=np.int64))

        # Edge points (label=1, near boundary).
        if self.max_edge_points > 0:
            edge_coords, num_edge = _sample_edge_points(
                mask, self.max_edge_points, band_width=10,
            )
            if num_edge > 0:
                edge_xy = edge_coords[:, [1, 0]].astype(np.float32)
                all_pts.append(edge_xy)
                all_labels.append(np.ones(num_edge, dtype=np.int64))

        pts = np.concatenate(all_pts, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        result = {
            'image': img_tensor,
            'mask': mask_tensor,
            'points': torch.from_numpy(pts),
            'labels': torch.from_numpy(labels),
        }

        # Bounding box prompt.
        if self.include_bbox:
            bbox = _mask_to_bbox(mask)  # [y1, x1, y2, x2]
            # Convert to SAM format: (x1, y1, x2, y2) normalized to image size.
            bbox_xyxy = np.array(
                [bbox[1], bbox[0], bbox[3], bbox[2]], dtype=np.float32,
            )
            result['bbox'] = torch.from_numpy(bbox_xyxy)

        return result


class FlatSegmentationDataset(Dataset):
    """PyTorch dataset for flat directory structures.

    Expects a layout like::

        data_dir/
            images/<name>.jpg  (or .png)
            masks/<name>.png
            train.txt          (one image stem per line)

    Each line in ``split_file`` is an image stem (e.g. ``frame_001``).
    The loader looks for ``images/{stem}.jpg`` (or ``.png``) and
    ``masks/{stem}.png``.

    Returns the same dictionary format as :class:`SegmentationDataset`.

    Args:
        data_dir: Root directory containing ``images/`` and ``masks/``.
        split_file: Path to a text file with one image stem per line.
        image_size: Spatial size for resizing.
        max_fg_points: Maximum foreground points per image.
        max_bg_points: Maximum background points per image.
        max_edge_points: Maximum edge points per image.
        augment: Whether to apply random augmentations.
        include_bbox: Whether to include bounding box prompts.
    """

    _IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        image_size: int = cfg.IMAGE_SIZE,
        max_fg_points: int = cfg.MAX_FG_POINTS,
        max_bg_points: int = cfg.MAX_BG_POINTS,
        max_edge_points: int = cfg.MAX_EDGE_POINTS,
        augment: bool = True,
        include_bbox: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.max_fg_points = max_fg_points
        self.max_bg_points = max_bg_points
        self.max_edge_points = max_edge_points
        self.augment = augment
        self.include_bbox = include_bbox

        data_dir = Path(data_dir)
        images_dir = data_dir / 'images'
        masks_dir = data_dir / 'masks'

        if not images_dir.is_dir():
            raise FileNotFoundError(f'Images directory not found: {images_dir}')
        if not masks_dir.is_dir():
            raise FileNotFoundError(f'Masks directory not found: {masks_dir}')

        # Build an index of available image files for fast lookup.
        image_index: Dict[str, Path] = {}
        for ext in self._IMAGE_EXTS:
            for p in images_dir.glob(f'*{ext}'):
                image_index[p.stem] = p

        self.samples: List[Tuple[str, str]] = []
        with open(split_file, 'r') as f:
            for line in f:
                stem = line.strip()
                if not stem or stem.startswith('#'):
                    continue

                # Resolve image path.
                img_path = image_index.get(stem)
                if img_path is None:
                    # Try stem without extension in case split file includes ext.
                    stem_no_ext = Path(stem).stem
                    img_path = image_index.get(stem_no_ext)
                    if img_path is None:
                        continue  # Skip missing images silently.

                # Resolve mask path.
                mask_path = masks_dir / (img_path.stem + '.png')
                if not mask_path.is_file():
                    continue  # Skip if mask is missing.

                self.samples.append((str(img_path), str(mask_path)))

        if not self.samples:
            raise RuntimeError(
                f'No valid samples found. split_file={split_file}, '
                f'images_dir={images_dir}, masks_dir={masks_dir}'
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[index]

        image = np.array(Image.open(img_path).convert('RGB'))
        mask_raw = np.array(Image.open(mask_path).convert('L'))
        mask = _binarize_mask(mask_raw, threshold=127)

        # Resize.
        image_pil = Image.fromarray(image).resize(
            (self.image_size, self.image_size), Image.BILINEAR,
        )
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST,
        )
        image = np.array(image_pil)
        mask = _binarize_mask(np.array(mask_pil), threshold=127)

        # Augmentation (training only).
        if self.augment:
            image, mask = _random_augment(image, mask)

        # Image tensor: (3, H, W) in [0, 1].
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        # Mask tensor: (1, H, W) in {0, 1}.
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # --- Point prompts ---
        all_pts = []
        all_labels = []

        # Foreground points (label=1).
        fg_coords, num_fg = _sample_foreground_points(mask, self.max_fg_points)
        fg_xy = fg_coords[:, [1, 0]].astype(np.float32)  # (row,col) -> (x,y)
        all_pts.append(fg_xy)
        all_labels.append(np.ones(num_fg, dtype=np.int64))

        # Background points (label=0).
        if self.max_bg_points > 0:
            bg_coords = _sample_background_points(
                mask, self.max_bg_points, margin=5,
            )
            bg_xy = bg_coords[:, [1, 0]].astype(np.float32)
            all_pts.append(bg_xy)
            all_labels.append(np.zeros(bg_coords.shape[0], dtype=np.int64))

        # Edge points (label=1, near boundary).
        if self.max_edge_points > 0:
            edge_coords, num_edge = _sample_edge_points(
                mask, self.max_edge_points, band_width=10,
            )
            if num_edge > 0:
                edge_xy = edge_coords[:, [1, 0]].astype(np.float32)
                all_pts.append(edge_xy)
                all_labels.append(np.ones(num_edge, dtype=np.int64))

        pts = np.concatenate(all_pts, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        result = {
            'image': img_tensor,
            'mask': mask_tensor,
            'points': torch.from_numpy(pts),
            'labels': torch.from_numpy(labels),
        }

        # Bounding box prompt.
        if self.include_bbox:
            bbox = _mask_to_bbox(mask)  # [y1, x1, y2, x2]
            # Convert to SAM format: (x1, y1, x2, y2) in pixel coordinates.
            bbox_xyxy = np.array(
                [bbox[1], bbox[0], bbox[3], bbox[2]], dtype=np.float32,
            )
            result['bbox'] = torch.from_numpy(bbox_xyxy)

        return result


class ClassificationSegmentationDataset(Dataset):
    """Dataset for classification + segmentation with class-specific masks.

    Expects a layout like::

        data_dir/
            images/<name>.jpg  (or .png)
            masks/<name>_cls<class_id>.png
            train.txt          (format: "name class_id" per line)
            class_map.json     (maps original class IDs to internal indices)

    Returns the same dictionary format as :class:`SegmentationDataset` plus
    ``class_label`` (one-hot tensor) and ``class_id`` (original class ID).

    Args:
        data_dir: Root directory containing ``images/`` and ``masks/``.
        split_file: Path to a text file with "name class_id" per line.
        class_map_path: Path to class_map.json.
        image_size: Spatial size for resizing.
        max_fg_points: Maximum foreground points per image.
        max_bg_points: Maximum background points per image.
        max_edge_points: Maximum edge points per image.
        augment: Whether to apply random augmentations.
        include_bbox: Whether to include bounding box prompts.
    """

    _IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        class_map_path: str,
        image_size: int = cfg.IMAGE_SIZE,
        max_fg_points: int = cfg.MAX_FG_POINTS,
        max_bg_points: int = cfg.MAX_BG_POINTS,
        max_edge_points: int = cfg.MAX_EDGE_POINTS,
        augment: bool = True,
        include_bbox: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.max_fg_points = max_fg_points
        self.max_bg_points = max_bg_points
        self.max_edge_points = max_edge_points
        self.augment = augment
        self.include_bbox = include_bbox

        data_dir = Path(data_dir)
        images_dir = data_dir / 'images'
        masks_dir = data_dir / 'masks'

        if not images_dir.is_dir():
            raise FileNotFoundError(f'Images directory not found: {images_dir}')
        if not masks_dir.is_dir():
            raise FileNotFoundError(f'Masks directory not found: {masks_dir}')

        # Load class map
        import json
        with open(class_map_path, 'r') as f:
            self.class_map = json.load(f)  # {original_id: internal_idx}
        self.num_classes = len(self.class_map)

        # Build an index of available image files for fast lookup.
        image_index: Dict[str, Path] = {}
        for ext in self._IMAGE_EXTS:
            for p in images_dir.glob(f'*{ext}'):
                image_index[p.stem] = p

        self.samples: List[Tuple[str, str, int]] = []
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                # Handle multiple class IDs (e.g., "0,1")
                class_id_str = parts[1].split(',')[0]
                class_id = int(class_id_str)

                # Resolve image path.
                img_path = image_index.get(name)
                if img_path is None:
                    continue

                # Resolve mask path (format: {name}_cls{class_id}.png)
                mask_path = masks_dir / f'{name}_cls{class_id}.png'
                if not mask_path.is_file():
                    # Try without class suffix
                    mask_path = masks_dir / f'{name}.png'
                    if not mask_path.is_file():
                        continue

                self.samples.append((str(img_path), str(mask_path), class_id))

        if not self.samples:
            raise RuntimeError(
                f'No valid samples found. split_file={split_file}, '
                f'images_dir={images_dir}, masks_dir={masks_dir}'
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path, class_id = self.samples[index]

        image = np.array(Image.open(img_path).convert('RGB'))
        mask_raw = np.array(Image.open(mask_path).convert('L'))
        mask = _binarize_mask(mask_raw, threshold=127)

        # Resize.
        image_pil = Image.fromarray(image).resize(
            (self.image_size, self.image_size), Image.BILINEAR,
        )
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST,
        )
        image = np.array(image_pil)
        mask = _binarize_mask(np.array(mask_pil), threshold=127)

        # Augmentation (training only).
        if self.augment:
            image, mask = _random_augment(image, mask)

        # Image tensor: (3, H, W) in [0, 1].
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        # Mask tensor: (1, H, W) in {0, 1}.
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # Class label (one-hot)
        internal_idx = self.class_map.get(str(class_id), 0)
        class_label = np.zeros(self.num_classes, dtype=np.float32)
        class_label[internal_idx] = 1.0

        # --- Point prompts ---
        all_pts = []
        all_labels = []

        # Foreground points (label=1).
        fg_coords, num_fg = _sample_foreground_points(mask, self.max_fg_points)
        fg_xy = fg_coords[:, [1, 0]].astype(np.float32)  # (row,col) -> (x,y)
        all_pts.append(fg_xy)
        all_labels.append(np.ones(num_fg, dtype=np.int64))

        # Background points (label=0).
        if self.max_bg_points > 0:
            bg_coords = _sample_background_points(
                mask, self.max_bg_points, margin=5,
            )
            bg_xy = bg_coords[:, [1, 0]].astype(np.float32)
            all_pts.append(bg_xy)
            all_labels.append(np.zeros(bg_coords.shape[0], dtype=np.int64))

        # Edge points (label=1, near boundary).
        if self.max_edge_points > 0:
            edge_coords, num_edge = _sample_edge_points(
                mask, self.max_edge_points, band_width=10,
            )
            if num_edge > 0:
                edge_xy = edge_coords[:, [1, 0]].astype(np.float32)
                all_pts.append(edge_xy)
                all_labels.append(np.ones(num_edge, dtype=np.int64))

        pts = np.concatenate(all_pts, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        result = {
            'image': img_tensor,
            'mask': mask_tensor,
            'points': torch.from_numpy(pts),
            'labels': torch.from_numpy(labels),
            'class_label': torch.from_numpy(class_label),
            'class_id': class_id,
        }

        # Bounding box prompt.
        if self.include_bbox:
            bbox = _mask_to_bbox(mask)  # [y1, x1, y2, x2]
            bbox_xyxy = np.array(
                [bbox[1], bbox[0], bbox[3], bbox[2]], dtype=np.float32,
            )
            result['bbox'] = torch.from_numpy(bbox_xyxy)

        return result


class InstanceSegmentationDataset(Dataset):
    """Dataset for instance-level segmentation with bbox annotations.

    Expects a layout like::

        data_dir/
            images/<name>.jpg  (or .png)
            masks/<name>.png
            labels/<name>.txt  (YOLO format: class_id cx cy w h, one per line)
            train.txt          (format: "name class_id" per line)
            class_map.json     (maps original class IDs to internal indices)

    Returns instance-level information including multiple bboxes and class labels
    for each defect in the image.

    Args:
        data_dir: Root directory containing ``images/``, ``masks/``, ``labels/``.
        split_file: Path to a text file with "name class_id" per line.
        class_map_path: Path to class_map.json.
        image_size: Spatial size for resizing.
        max_fg_points: Maximum foreground points per image.
        max_bg_points: Maximum background points per image.
        max_edge_points: Maximum edge points per image.
        augment: Whether to apply random augmentations.
        include_bbox: Whether to include bounding box prompts.
    """

    _IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        class_map_path: str,
        image_size: int = cfg.IMAGE_SIZE,
        max_fg_points: int = cfg.MAX_FG_POINTS,
        max_bg_points: int = cfg.MAX_BG_POINTS,
        max_edge_points: int = cfg.MAX_EDGE_POINTS,
        augment: bool = True,
        include_bbox: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.max_fg_points = max_fg_points
        self.max_bg_points = max_bg_points
        self.max_edge_points = max_edge_points
        self.augment = augment
        self.include_bbox = include_bbox

        data_dir = Path(data_dir)
        images_dir = data_dir / 'images'
        masks_dir = data_dir / 'masks'
        labels_dir = data_dir / 'labels'

        if not images_dir.is_dir():
            raise FileNotFoundError(f'Images directory not found: {images_dir}')
        if not masks_dir.is_dir():
            raise FileNotFoundError(f'Masks directory not found: {masks_dir}')
        if not labels_dir.is_dir():
            raise FileNotFoundError(f'Labels directory not found: {labels_dir}')

        # Load class map
        import json
        with open(class_map_path, 'r') as f:
            self.class_map = json.load(f)  # {original_id: internal_idx}
        self.num_classes = len(self.class_map)

        # Build an index of available image files for fast lookup.
        image_index: Dict[str, Path] = {}
        for ext in self._IMAGE_EXTS:
            for p in images_dir.glob(f'*{ext}'):
                image_index[p.stem] = p

        self.samples: List[Tuple[str, str, str, int]] = []
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                # Handle multiple class IDs (e.g., "0,1")
                class_id_str = parts[1].split(',')[0]
                class_id = int(class_id_str)

                # Resolve image path.
                img_path = image_index.get(name)
                if img_path is None:
                    continue

                # Resolve mask path
                mask_path = masks_dir / f'{name}.png'
                if not mask_path.is_file():
                    continue

                # Resolve label path (bbox annotations)
                label_path = labels_dir / f'{name}.txt'
                if not label_path.is_file():
                    continue

                self.samples.append((str(img_path), str(mask_path), str(label_path), class_id))

        if not self.samples:
            raise RuntimeError(
                f'No valid samples found. split_file={split_file}, '
                f'images_dir={images_dir}, masks_dir={masks_dir}, labels_dir={labels_dir}'
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path, label_path, class_id = self.samples[index]

        image = np.array(Image.open(img_path).convert('RGB'))
        mask_raw = np.array(Image.open(mask_path).convert('L'))
        mask = _binarize_mask(mask_raw, threshold=127)

        # Resize.
        image_pil = Image.fromarray(image).resize(
            (self.image_size, self.image_size), Image.BILINEAR,
        )
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST,
        )
        image = np.array(image_pil)
        mask = _binarize_mask(np.array(mask_pil), threshold=127)

        # Augmentation (training only).
        if self.augment:
            image, mask = _random_augment(image, mask)

        # Image tensor: (3, H, W) in [0, 1].
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        # Mask tensor: (1, H, W) in {0, 1}.
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # Load bbox annotations (YOLO format: class_id cx cy w h)
        bboxes = []
        bbox_class_ids = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    # Convert YOLO format to x1, y1, x2, y2 (normalized)
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    bboxes.append([x1, y1, x2, y2])
                    bbox_class_ids.append(cls_id)

        # If no bboxes found, use mask to generate one
        if not bboxes:
            bbox = _mask_to_bbox(mask)  # [y1, x1, y2, x2]
            bbox_xyxy = np.array([bbox[1], bbox[0], bbox[3], bbox[2]], dtype=np.float32)
            # Normalize
            bbox_xyxy[0] /= self.image_size
            bbox_xyxy[1] /= self.image_size
            bbox_xyxy[2] /= self.image_size
            bbox_xyxy[3] /= self.image_size
            bboxes = [bbox_xyxy.tolist()]
            bbox_class_ids = [class_id]

        # Class label (image-level, multi-hot)
        internal_idx = self.class_map.get(str(class_id), 0)
        class_label = np.zeros(self.num_classes, dtype=np.float32)
        class_label[internal_idx] = 1.0

        # Instance-level class labels (one per bbox)
        instance_class_labels = []
        for cls_id in bbox_class_ids:
            idx = self.class_map.get(str(cls_id), 0)
            instance_class_labels.append(idx)

        # --- Point prompts ---
        all_pts = []
        all_labels = []

        # Foreground points (label=1).
        fg_coords, num_fg = _sample_foreground_points(mask, self.max_fg_points)
        fg_xy = fg_coords[:, [1, 0]].astype(np.float32)  # (row,col) -> (x,y)
        all_pts.append(fg_xy)
        all_labels.append(np.ones(num_fg, dtype=np.int64))

        # Background points (label=0).
        if self.max_bg_points > 0:
            bg_coords = _sample_background_points(
                mask, self.max_bg_points, margin=5,
            )
            bg_xy = bg_coords[:, [1, 0]].astype(np.float32)
            all_pts.append(bg_xy)
            all_labels.append(np.zeros(bg_coords.shape[0], dtype=np.int64))

        # Edge points (label=1, near boundary).
        if self.max_edge_points > 0:
            edge_coords, num_edge = _sample_edge_points(
                mask, self.max_edge_points, band_width=10,
            )
            if num_edge > 0:
                edge_xy = edge_coords[:, [1, 0]].astype(np.float32)
                all_pts.append(edge_xy)
                all_labels.append(np.ones(num_edge, dtype=np.int64))

        pts = np.concatenate(all_pts, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        result = {
            'image': img_tensor,
            'mask': mask_tensor,
            'points': torch.from_numpy(pts),
            'labels': torch.from_numpy(labels),
            'class_label': torch.from_numpy(class_label),
            'class_id': class_id,
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),  # (N, 4) normalized
            'instance_class_labels': torch.tensor(instance_class_labels, dtype=torch.long),  # (N,)
        }

        return result


class BboxPointPromptDataset(Dataset):
    """Dataset with bbox-based point prompts for SAM2 training.

    Expects a layout like::

        data_dir/
            images/<name>.jpg  (or .png)
            masks/<name>.png
            labels/<name>.txt  (YOLO format: class_id cx cy w h)
            points/<name>.txt  (format: label x y, one per line, normalized)
            train.txt          (format: "name class_id" per line)
            class_map.json     (maps original class IDs to internal indices)

    Uses pre-computed point prompts from bbox annotations instead of random sampling.

    Args:
        data_dir: Root directory containing images/, masks/, labels/, points/.
        split_file: Path to a text file with "name class_id" per line.
        class_map_path: Path to class_map.json.
        image_size: Spatial size for resizing.
        augment: Whether to apply random augmentations.
    """

    _IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

    def __init__(
        self,
        data_dir: str,
        split_file: str,
        class_map_path: str,
        image_size: int = cfg.IMAGE_SIZE,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.augment = augment

        data_dir = Path(data_dir)
        images_dir = data_dir / 'images'
        masks_dir = data_dir / 'masks'
        points_dir = data_dir / 'points'

        if not images_dir.is_dir():
            raise FileNotFoundError(f'Images directory not found: {images_dir}')
        if not masks_dir.is_dir():
            raise FileNotFoundError(f'Masks directory not found: {masks_dir}')
        if not points_dir.is_dir():
            raise FileNotFoundError(f'Points directory not found: {points_dir}')

        # Load class map
        import json
        with open(class_map_path, 'r') as f:
            self.class_map = json.load(f)
        self.num_classes = len(self.class_map)

        # Build an index of available image files for fast lookup.
        image_index: Dict[str, Path] = {}
        for ext in self._IMAGE_EXTS:
            for p in images_dir.glob(f'*{ext}'):
                image_index[p.stem] = p

        self.samples: List[Tuple[str, str, str, int]] = []
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                class_id_str = parts[1].split(',')[0]
                class_id = int(class_id_str)

                img_path = image_index.get(name)
                if img_path is None:
                    continue

                mask_path = masks_dir / f'{name}.png'
                if not mask_path.is_file():
                    continue

                pts_path = points_dir / f'{name}.txt'
                if not pts_path.is_file():
                    continue

                self.samples.append((str(img_path), str(mask_path), str(pts_path), class_id))

        if not self.samples:
            raise RuntimeError(
                f'No valid samples found. split_file={split_file}, '
                f'images_dir={images_dir}, masks_dir={masks_dir}, points_dir={points_dir}'
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path, pts_path, class_id = self.samples[index]

        image = np.array(Image.open(img_path).convert('RGB'))
        mask_raw = np.array(Image.open(mask_path).convert('L'))
        mask = _binarize_mask(mask_raw, threshold=127)

        # Resize
        image_pil = Image.fromarray(image).resize(
            (self.image_size, self.image_size), Image.BILINEAR,
        )
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST,
        )
        image = np.array(image_pil)
        mask = _binarize_mask(np.array(mask_pil), threshold=127)

        # Augmentation
        if self.augment:
            image, mask = _random_augment(image, mask)

        # Image tensor: (3, H, W) in [0, 1]
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        # Mask tensor: (1, H, W) in {0, 1}
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        # Load point prompts from file (normalized coordinates)
        all_pts = []
        all_labels = []
        with open(pts_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    label = int(parts[0])
                    x = float(parts[1])  # normalized [0, 1]
                    y = float(parts[2])  # normalized [0, 1]
                    # Convert to pixel coordinates
                    px = x * self.image_size
                    py = y * self.image_size
                    all_pts.append([px, py])
                    all_labels.append(label)

        # If no points found, use center point
        if not all_pts:
            all_pts = [[self.image_size / 2, self.image_size / 2]]
            all_labels = [1]

        pts = np.array(all_pts, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int64)

        # Class label (image-level, multi-hot)
        internal_idx = self.class_map.get(str(class_id), 0)
        class_label = np.zeros(self.num_classes, dtype=np.float32)
        class_label[internal_idx] = 1.0

        result = {
            'image': img_tensor,
            'mask': mask_tensor,
            'points': torch.from_numpy(pts),
            'labels': torch.from_numpy(labels),
            'class_label': torch.from_numpy(class_label),
            'class_id': class_id,
        }

        return result


def pad_collate(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate function that pads point coordinates and labels to equal length.

    SAM2 requires a fixed number of points in each batch element.  This
    function zero-pads the point tensor and uses label ``-1`` to indicate
    padding positions (ignored by the prompt encoder).

    Args:
        batch: List of samples from ``SegmentationDataset``.

    Returns:
        Batched dictionary with keys ``image``, ``mask``, ``points``,
        ``labels``, and optionally ``bbox``.
    """
    images = torch.stack([b['image'] for b in batch])
    masks = torch.stack([b['mask'] for b in batch])

    max_n = max(b['points'].shape[0] for b in batch)
    padded_pts = []
    padded_lbls = []
    for b in batch:
        n = b['points'].shape[0]
        pad_len = max_n - n
        if pad_len > 0:
            pts = torch.cat([
                b['points'],
                torch.zeros(pad_len, 2, dtype=torch.float),
            ], dim=0)
            lbls = torch.cat([
                b['labels'],
                -torch.ones(pad_len, dtype=torch.long),
            ], dim=0)
        else:
            pts = b['points']
            lbls = b['labels']
        padded_pts.append(pts)
        padded_lbls.append(lbls)

    result = {
        'image': images,
        'mask': masks,
        'points': torch.stack(padded_pts),
        'labels': torch.stack(padded_lbls),
    }

    if 'bbox' in batch[0]:
        result['bbox'] = torch.stack([b['bbox'] for b in batch])

    if 'class_label' in batch[0]:
        result['class_label'] = torch.stack([b['class_label'] for b in batch])
        result['class_id'] = [b['class_id'] for b in batch]

    if 'bboxes' in batch[0]:
        # Pad bboxes and instance_class_labels to equal length
        max_bboxes = max(b['bboxes'].shape[0] for b in batch)
        padded_bboxes = []
        padded_instance_cls = []
        for b in batch:
            n = b['bboxes'].shape[0]
            pad_len = max_bboxes - n
            if pad_len > 0:
                bbox = torch.cat([
                    b['bboxes'],
                    torch.zeros(pad_len, 4, dtype=torch.float),
                ], dim=0)
                cls = torch.cat([
                    b['instance_class_labels'],
                    -torch.ones(pad_len, dtype=torch.long),
                ], dim=0)
            else:
                bbox = b['bboxes']
                cls = b['instance_class_labels']
            padded_bboxes.append(bbox)
            padded_instance_cls.append(cls)
        result['bboxes'] = torch.stack(padded_bboxes)
        result['instance_class_labels'] = torch.stack(padded_instance_cls)

    return result
