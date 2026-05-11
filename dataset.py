"""
dataset.py — PyTorch Dataset classes for HW3 Instance Segmentation
on H&E medical images.

Classes
-------
CellDataset   : Training dataset that parses mask .tif files and returns
                (image, target) pairs compatible with Mask R-CNN.
TestDataset   : Inference dataset that returns (image, image_id) pairs.

Helpers
-------
get_train_val_split : Stratified train/val split via sklearn.
collate_fn          : Variable-size collate for DataLoader.

Usage example (albumentations plug-in)
---------------------------------------
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For training you would wrap albumentations yourself, e.g.:
#   def my_transform(image, target): ...
# then pass it as `transforms=my_transform`.
# For test, transforms(image) must return a numpy array or Tensor.
# Default is None — raw FloatTensor[3,H,W] in [0,1] is returned.
"""

import json
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import skimage.io as skio
from skimage.transform import resize as _sk_resize

# ---------------------------------------------------------------------------
# Optional imagecodecs import — required for LZW-compressed image.tif files.
# Mask .tif files are uncompressed and work without this package.
# ---------------------------------------------------------------------------
try:
    import imagecodecs  # noqa: F401 — side-effect: registers LZW codec
    _IMAGECODECS_OK = True
except ImportError:
    _IMAGECODECS_OK = False
    warnings.warn(
        "imagecodecs not found. RGB image.tif (LZW-compressed) files may "
        "fail to read. Install with: pip install imagecodecs",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CLASS_NAMES = ["class1", "class2", "class3", "class4"]
_NUM_CLASSES = len(_CLASS_NAMES)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _read_image(path: Path) -> np.ndarray:
    """Read an RGB(A) .tif image and return a uint8 array of shape (H,W,3).

    Parameters
    ----------
    path : Path
        Absolute path to the .tif file.

    Returns
    -------
    np.ndarray
        uint8 array shaped (H, W, 3).

    Raises
    ------
    IOError
        If the file cannot be read by skimage.
    """
    img = skio.imread(str(path))
    img = np.asarray(img)

    if img.ndim == 2:
        # Grayscale — stack into 3-channel image
        img = np.stack([img, img, img], axis=-1)

    if img.ndim == 3 and img.shape[2] > 3:
        # RGBA or other multi-channel — keep only first 3 channels
        img = img[:, :, :3]

    if img.dtype != np.uint8:
        # Normalise to uint8 for consistency
        if img.dtype == np.uint16:
            img = (img / 257).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img


def _read_mask(path: Path) -> np.ndarray:
    """Read a mask .tif file and return a 2-D integer array.

    Pixel values encode instance IDs; background == 0.

    Parameters
    ----------
    path : Path
        Absolute path to the mask .tif file.

    Returns
    -------
    np.ndarray
        Integer array shaped (H, W).
    """
    mask = skio.imread(str(path))
    mask = np.asarray(mask)

    if mask.ndim == 3:
        # Some encoders save grayscale as (H, W, 1) or (H, W, C)
        mask = mask[:, :, 0]

    # Cast float64 instance IDs to int for safe comparison
    return mask.astype(np.int64)


def _parse_instances(uuid_dir: Path, max_side: int = 0):
    """Parse all classN.tif masks in *uuid_dir* into per-instance data.

    Parameters
    ----------
    uuid_dir : Path
        Directory containing classN.tif files for one training sample.
    max_side : int
        If > 0 and max(H, W) exceeds this value, each class mask is
        downscaled with nearest-neighbour interpolation before instance
        extraction to prevent RAM OOM on large images.  0 = no limit.

    Returns
    -------
    tuple
        (boxes, labels, masks_list, H, W) where
        - boxes      : list of [x_min, y_min, x_max, y_max] (xyxy, float)
        - labels     : list of int (1-indexed class IDs)
        - masks_list : list of bool np.ndarray shaped (H, W)
        - H, W       : spatial dimensions (possibly downscaled)
    """
    boxes = []
    labels = []
    masks_list = []
    H = W = None
    rH = rW = None  # downscaled target dims; None means no downscale

    for cls_idx, cls_name in enumerate(_CLASS_NAMES):
        mask_path = uuid_dir / f"{cls_name}.tif"
        if not mask_path.exists():
            continue

        mask = _read_mask(mask_path)

        if H is None:
            H, W = mask.shape[:2]
            if max_side > 0 and max(H, W) > max_side:
                scale = max_side / max(H, W)
                rH = int(round(H * scale))
                rW = int(round(W * scale))

        # Downscale class mask if needed; nearest-neighbour preserves IDs.
        if rH is not None:
            mask = _sk_resize(
                mask, (rH, rW), order=0,
                preserve_range=True, anti_aliasing=False,
            ).astype(np.int64)

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]  # drop background

        for iid in instance_ids:
            iid_int = int(iid)
            binary = (mask == iid_int)

            rows, cols = np.where(binary)
            if rows.size == 0:
                continue

            x_min = int(cols.min())
            y_min = int(rows.min())
            x_max = int(cols.max()) + 1  # exclusive end → xyxy convention
            y_max = int(rows.max()) + 1

            # Skip degenerate boxes with zero area
            if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(cls_idx + 1)   # 1-indexed
            masks_list.append(binary)

    out_H = rH if rH is not None else H
    out_W = rW if rW is not None else W
    return boxes, labels, masks_list, out_H, out_W


# ===========================================================================
# CellDataset
# ===========================================================================

class CellDataset(Dataset):
    """PyTorch Dataset for H&E cell instance segmentation (training).

    Reads RGB images and corresponding instance mask files from the
    ``hw3-data-release/train/`` directory structure.

    Parameters
    ----------
    root_dir : str or Path
        Path to the ``train/`` directory that contains UUID sub-directories.
    transforms : callable or None
        Optional transform applied as ``transforms(image, target)``.
        *image* is a ``FloatTensor[3, H, W]`` in ``[0, 1]``;
        *target* is the dict described in ``__getitem__``.
        If ``None`` the raw tensors are returned unchanged.
        To plug in albumentations, wrap it in a custom callable that
        converts back to tensors after augmentation.
    use_cache : bool
        If ``True``, parsed targets (masks, boxes, labels) are cached in
        memory after the first access. Speeds up repeated epoch scans at
        the cost of RAM (~few hundred MB for the full dataset).

    Notes
    -----
    * Requires ``imagecodecs`` to be installed for reading LZW-compressed
      ``image.tif`` files.  Mask files do not need it.
    * Images with *no* annotated instances are included; their target
      tensors are empty (shape ``[0, ...]``) rather than being dropped.
    """

    def __init__(
        self,
        root_dir,
        transforms=None,
        use_cache: bool = False,
        max_side: int = 1333,
    ):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.use_cache = use_cache
        self.max_side = max_side  # 0 = no limit; matches torchvision default max_size

        # Collect all UUID subdirectories that contain at least image.tif
        self.samples = sorted(
            p for p in self.root_dir.iterdir()
            if p.is_dir()
        )

        if len(self.samples) == 0:
            warnings.warn(
                f"No sample directories found under {self.root_dir}",
                stacklevel=2,
            )

        # Cache dict: idx -> target dict (stored as numpy, converted on get)
        self._cache: dict = {}

    def __len__(self) -> int:
        """Return the number of training samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return a single training sample.

        Parameters
        ----------
        idx : int
            Index into the dataset.

        Returns
        -------
        image : FloatTensor[3, H, W]
            Normalised RGB image in ``[0.0, 1.0]``.
        target : dict
            ``"boxes"``    : FloatTensor[N, 4] — xyxy bounding boxes
            ``"labels"``   : Int64Tensor[N]    — 1-indexed class IDs
            ``"masks"``    : BoolTensor[N, H, W]
            ``"image_id"`` : Int64Tensor[1]    — equals *idx*
            ``"area"``     : FloatTensor[N]    — bbox w*h
            ``"iscrowd"``  : Int64Tensor[N]    — always 0
        """
        uuid_dir = self.samples[idx]

        # ---- Load image -------------------------------------------------- #
        img_np = _read_image(uuid_dir / "image.tif")
        H, W = img_np.shape[:2]

        # Downscale large images so the mask stack stays within RAM limits.
        # Must match the resize applied inside _parse_instances.
        if self.max_side > 0 and max(H, W) > self.max_side:
            scale = self.max_side / max(H, W)
            rH = int(round(H * scale))
            rW = int(round(W * scale))
            img_np = _sk_resize(
                img_np, (rH, rW), order=1,
                preserve_range=True, anti_aliasing=True,
            ).astype(np.uint8)
            H, W = rH, rW

        image = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        # ---- Load / cache target ----------------------------------------- #
        if self.use_cache and idx in self._cache:
            # Clone every tensor so that in-place mutations by transforms
            # cannot corrupt the cached originals.
            target = {k: v.clone() for k, v in self._cache[idx].items()}
        else:
            target = self._build_target(uuid_dir, idx, H, W)
            if self.use_cache:
                # Store clones so the caller's reference is independent.
                self._cache[idx] = {k: v.clone() for k, v in target.items()}

        # ---- Optional transforms ----------------------------------------- #
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _build_target(
        self,
        uuid_dir: Path,
        idx: int,
        H: int,
        W: int,
    ) -> dict:
        """Build the target dict for one sample.

        Parameters
        ----------
        uuid_dir : Path
            Sample directory.
        idx : int
            Dataset index (used as image_id).
        H, W : int
            Image spatial dimensions, used for empty-mask fallback.

        Returns
        -------
        dict
            Target dict with tensors as described in ``__getitem__``.
        """
        boxes_raw, labels_raw, masks_raw, mask_H, mask_W = (
            _parse_instances(uuid_dir, max_side=self.max_side)
        )

        # Fallback if no mask files found: use image dimensions
        if mask_H is None:
            mask_H, mask_W = H, W

        if len(labels_raw) == 0:
            # No annotated instances — return empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros(
                (0, mask_H, mask_W), dtype=torch.bool
            )
            area = torch.zeros(0, dtype=torch.float32)
            iscrowd = torch.zeros(0, dtype=torch.int64)
        else:
            boxes_np = np.array(boxes_raw, dtype=np.float32)  # (N,4)
            labels_np = np.array(labels_raw, dtype=np.int64)  # (N,)
            masks_np = np.stack(masks_raw, axis=0)             # (N,H,W)

            boxes = torch.from_numpy(boxes_np)
            labels = torch.from_numpy(labels_np)
            masks = torch.from_numpy(masks_np)

            # bbox area = w * h  (xyxy format)
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            area = (widths * heights).to(torch.float32)
            iscrowd = torch.zeros(len(labels_raw), dtype=torch.int64)

        return {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

    def get_class_sets(self) -> List[frozenset]:
        """Return the frozenset of present classes for each sample.

        Used by :func:`get_train_val_split` for stratification.

        Returns
        -------
        list of frozenset
            One frozenset per sample; each element is a 1-based class ID.
        """
        class_sets = []
        for uuid_dir in self.samples:
            present = frozenset(
                cls_idx + 1
                for cls_idx, cls_name in enumerate(_CLASS_NAMES)
                if (uuid_dir / f"{cls_name}.tif").exists()
            )
            if not present:
                present = frozenset({0})   # sentinel for "no class"
            class_sets.append(present)
        return class_sets


# ===========================================================================
# TestDataset
# ===========================================================================

class TestDataset(Dataset):
    """PyTorch Dataset for H&E cell instance segmentation (inference).

    Reads test images from a flat directory and pairs each with its
    integer image ID from the accompanying JSON file.

    Parameters
    ----------
    test_dir : str or Path
        Path to the ``test_release/`` directory.
    json_path : str or Path
        Path to ``test_image_name_to_ids.json``.
    transforms : callable or None
        Optional transform applied as ``transforms(image)`` where *image*
        is a ``FloatTensor[3, H, W]`` in ``[0, 1]``.
        Return type must also be a ``FloatTensor[3, H, W]``.
    """

    def __init__(
        self,
        test_dir,
        json_path,
        transforms: Optional[Callable] = None,
    ):
        self.test_dir = Path(test_dir)
        self.transforms = transforms

        with open(json_path, "r") as fh:
            entries = json.load(fh)

        # Build list of (absolute_path, image_id) pairs in JSON order
        self.samples = []
        for entry in entries:
            img_path = self.test_dir / entry["file_name"]
            self.samples.append((img_path, int(entry["id"])))

        if len(self.samples) == 0:
            warnings.warn(
                f"No test entries found in {json_path}",
                stacklevel=2,
            )

    def __len__(self) -> int:
        """Return the number of test images."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return one test sample.

        Parameters
        ----------
        idx : int
            Index into the dataset.

        Returns
        -------
        image : FloatTensor[3, H, W]
            Normalised RGB image in ``[0.0, 1.0]``.
        image_id : int
            Integer ID from the JSON metadata.
        """
        img_path, image_id = self.samples[idx]

        img_np = _read_image(img_path)
        image = (
            torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        )

        if self.transforms is not None:
            image = self.transforms(image)

        return image, image_id


# ===========================================================================
# get_train_val_split
# ===========================================================================

def get_train_val_split(
    dataset: CellDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """Stratified train / val split of a :class:`CellDataset`.

    Stratification is based on the *set of classes* present in each
    image, so that every combination of co-occurring classes is
    represented proportionally in both splits.

    Parameters
    ----------
    dataset : CellDataset
        The full training dataset.
    val_ratio : float
        Fraction of samples to place in the validation split (default 0.2).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    train_subset : torch.utils.data.Subset
    val_subset   : torch.utils.data.Subset

    Notes
    -----
    Requires ``scikit-learn``.  If a stratum contains fewer than 2
    samples, ``sklearn`` will raise a ``ValueError``; in that case the
    split falls back to a random (non-stratified) split with a warning.
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for get_train_val_split. "
            "Install with: pip install scikit-learn"
        ) from exc

    indices = list(range(len(dataset)))
    class_sets = dataset.get_class_sets()

    # Encode frozensets as strings for sklearn stratify parameter
    strata = [str(sorted(cs)) for cs in class_sets]

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            stratify=strata,
        )
    except ValueError as exc:
        warnings.warn(
            f"Stratified split failed ({exc}). "
            "Falling back to random split.",
            stacklevel=2,
        )
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            stratify=None,
        )

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ===========================================================================
# collate_fn
# ===========================================================================

def collate_fn(batch: List[Tuple]) -> Tuple[Tuple, Tuple]:
    """Collate function for variable-size images and targets.

    Zips a list of ``(image, target)`` pairs into two tuples so that
    each tuple can be passed directly to a Mask R-CNN model.

    Parameters
    ----------
    batch : list of tuple
        Output of ``CellDataset.__getitem__`` repeated *batch_size* times.

    Returns
    -------
    tuple
        ``(images_tuple, targets_tuple)``
    """
    return tuple(zip(*batch))


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import sys

    # Force UTF-8 output on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  dataset.py — self-test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. CellDataset smoke test
    # ------------------------------------------------------------------
    TRAIN_DIR = "hw3-data-release/train"
    print(f"\nLoading CellDataset from: {TRAIN_DIR}")
    dataset = CellDataset(TRAIN_DIR)
    print(f"Dataset length: {len(dataset)}")

    img, target = dataset[0]
    print("\n[Sample 0]")
    print(f"  Image  : {img.shape}, dtype={img.dtype}, "
          f"min={img.min():.3f}, max={img.max():.3f}")
    print(f"  Instances : {len(target['labels'])}")
    print(f"  Labels[:10]: {target['labels'][:10].tolist()}")
    print(f"  Boxes[:3] :\n{target['boxes'][:3]}")
    print(f"  Masks shape: {target['masks'].shape}, "
          f"dtype={target['masks'].dtype}")
    print(f"  Area[:3]  : {target['area'][:3].tolist()}")
    print(f"  image_id  : {target['image_id'].item()}")
    print(f"  iscrowd[:3]: {target['iscrowd'][:3].tolist()}")

    # ------------------------------------------------------------------
    # 2. Train / val split
    # ------------------------------------------------------------------
    print("\nComputing stratified train/val split (val_ratio=0.2) ...")
    train_ds, val_ds = get_train_val_split(dataset)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    assert len(train_ds) + len(val_ds) == len(dataset), (
        "Split sizes do not add up to dataset size!"
    )

    # ------------------------------------------------------------------
    # 3. TestDataset smoke test
    # ------------------------------------------------------------------
    TEST_DIR = "hw3-data-release/test_release"
    TEST_JSON = "hw3-data-release/test_image_name_to_ids.json"
    print(f"\nLoading TestDataset from: {TEST_DIR}")
    test_ds = TestDataset(TEST_DIR, TEST_JSON)
    print(f"TestDataset length: {len(test_ds)}")

    t_img, t_id = test_ds[0]
    print("[Test sample 0]")
    print(f"  Image  : {t_img.shape}, dtype={t_img.dtype}, "
          f"min={t_img.min():.3f}, max={t_img.max():.3f}")
    print(f"  image_id: {t_id}")

    # ------------------------------------------------------------------
    # 4. collate_fn smoke test
    # ------------------------------------------------------------------
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    images, targets = collate_fn(batch)
    print(f"\ncollate_fn: batch of {len(images)} images OK")

    # ------------------------------------------------------------------
    # 5. Cache test
    # ------------------------------------------------------------------
    print("\nTesting use_cache=True ...")
    ds_cached = CellDataset(TRAIN_DIR, use_cache=True)
    img_c, tgt_c = ds_cached[0]   # first access — populates cache
    img_c2, tgt_c2 = ds_cached[0]  # second access — from cache
    assert torch.equal(img_c, img_c2), "Cached image mismatch!"
    assert torch.equal(
        tgt_c["boxes"], tgt_c2["boxes"]
    ), "Cached boxes mismatch!"
    print("  Cache hit verified.")

    print("\n" + "=" * 60)
    print("  All self-tests passed.")
    print("=" * 60)
