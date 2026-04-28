"""
train.py — Training script for HW3 Instance Segmentation on H&E images.

Usage
-----
python train.py [--data-dir PATH] [--epochs N] [--batch-size N]
                [--lr F] [--resume PATH] [--num-workers N]
                [--device cuda|cpu]

Smoke test
----------
python train.py --epochs 1 --batch-size 1 --num-workers 0
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- Windows console UTF-8 fix --------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- torchvision imports --------------------------------------------------
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ---- pycocotools -----------------------------------------------------------
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

# ---- local -----------------------------------------------------------------
from dataset import (
    CellDataset,
    get_train_val_split,
    collate_fn,
)
import random
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter as _ColorJitter

# ===========================================================================
# encode_mask helper (inlined from sample_code/utils.py)
# ===========================================================================


def encode_mask(binary_mask):
    """Encode a binary numpy mask to COCO RLE format.

    Parameters
    ----------
    binary_mask : np.ndarray
        Bool or uint8 array of shape (H, W).

    Returns
    -------
    dict
        RLE dict with 'counts' as a UTF-8 string (JSON-serialisable).
    """
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


class TrainTransform:
    """Joint image + target augmentation applied to the training set only."""

    def __init__(self):
        self.color_jitter = _ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0
        )

    def __call__(self, image, target):
        _, H, W = image.shape

        if random.random() < 0.5:                        # random H-flip
            image = TF.hflip(image)
            target["masks"] = target["masks"].flip(-1)
            boxes = target["boxes"].clone()
            boxes[:, 0] = W - target["boxes"][:, 2]
            boxes[:, 2] = W - target["boxes"][:, 0]
            target["boxes"] = boxes

        if random.random() < 0.5:                        # random V-flip
            image = TF.vflip(image)
            target["masks"] = target["masks"].flip(-2)
            boxes = target["boxes"].clone()
            boxes[:, 1] = H - target["boxes"][:, 3]
            boxes[:, 3] = H - target["boxes"][:, 1]
            target["boxes"] = boxes

        image = self.color_jitter(image)                 # colour only, no mask change
        return image, target


# ===========================================================================
# Model builder
# ===========================================================================


def build_model(num_classes: int = 5) -> nn.Module:
    """Build Mask R-CNN v2 with COCO pretrained weights, fine-tuned heads.

    Loads maskrcnn_resnet50_fpn_v2 with full COCO pretrained weights, then
    replaces the box predictor and mask predictor heads to match num_classes.
    The RPN anchor generator is replaced with EDA-tuned sizes for H&E cells.

    Parameters
    ----------
    num_classes : int
        Total number of classes including background (default 5).

    Returns
    -------
    nn.Module
        The model (not yet moved to a device).
    """
    # Load full COCO pretrained model (91 classes)
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        box_detections_per_img=800,
        rpn_post_nms_top_n_train=3000,
        rpn_post_nms_top_n_test=1000,
        image_mean=[0.6747, 0.5388, 0.7283],
        image_std=[0.1763, 0.2312, 0.1934],
    )

    # Replace box predictor head (91 → num_classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor head (91 → num_classes)
    in_ch = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_ch, 256, num_classes)

    # Replace with EDA-tuned anchors for H&E cell sizes
    anchor_generator = AnchorGenerator(
        sizes=((19,), (23,), (31,), (53,), (93,)),
        aspect_ratios=((0.7, 1.0, 1.4),) * 5,
    )
    model.rpn.anchor_generator = anchor_generator

    return model


# ===========================================================================
# COCO ground-truth builder
# ===========================================================================


def build_coco_gt(val_subset) -> dict:
    """Build a COCO ground-truth dict from the validation subset.

    Parameters
    ----------
    val_subset : torch.utils.data.Subset
        Validation subset returned by get_train_val_split.

    Returns
    -------
    dict
        COCO-format dict with 'images', 'annotations', 'categories'.
    """
    images = []
    annotations = []
    ann_id = 1

    for i, global_idx in enumerate(val_subset.indices):
        # image_id == global_idx (as stored in target["image_id"])
        img_id = global_idx

        img_tensor, target = val_subset.dataset[global_idx]

        # img_tensor: FloatTensor[3, H, W]
        _, img_H, img_W = img_tensor.shape

        images.append(
            {"id": img_id, "height": img_H, "width": img_W}
        )

        boxes = target["boxes"]       # FloatTensor[N,4] xyxy
        labels = target["labels"]     # Int64Tensor[N]
        masks = target["masks"]       # BoolTensor[N,H,W]
        iscrowd = target["iscrowd"]

        num_inst = len(labels)
        for k in range(num_inst):
            x1, y1, x2, y2 = boxes[k].tolist()
            w = x2 - x1
            h = y2 - y1
            binary = masks[k].numpy()
            rle = encode_mask(binary)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(labels[k].item()),
                    "bbox": [x1, y1, w, h],
                    "area": float(w * h),
                    "iscrowd": int(iscrowd[k].item()),
                    "segmentation": rle,
                }
            )
            ann_id += 1

    categories = [
        {"id": i, "name": f"class{i}"} for i in range(1, 5)
    ]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ===========================================================================
# Move targets to device
# ===========================================================================


def targets_to_device(targets, device):
    """Move all tensors in a list of target dicts to *device*.

    Parameters
    ----------
    targets : list of dict
    device : torch.device

    Returns
    -------
    list of dict
    """
    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v
         for k, v in t.items()}
        for t in targets
    ]


# ===========================================================================
# Training one epoch
# ===========================================================================


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    total_epochs: int,
    grad_clip: float = 5.0,
):
    """Run one training epoch.

    Parameters
    ----------
    model : nn.Module
        Mask R-CNN model in training mode.
    loader : DataLoader
    optimizer : torch.optim.Optimizer
    scaler : torch.cuda.amp.GradScaler
    device : torch.device
    epoch : int
        Current epoch index (1-based for display).
    total_epochs : int
    grad_clip : float

    Returns
    -------
    dict
        Mean loss values over the epoch:
        loss_total, loss_classifier, loss_box_reg, loss_mask,
        loss_objectness, loss_rpn_box_reg.
    """
    model.train()

    sum_losses = {
        "loss_total": 0.0,
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_mask": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    n_batches = 0

    desc = f"Epoch {epoch:02d}/{total_epochs:02d} [Train]"
    pbar = tqdm(loader, desc=desc, leave=True)

    for imgs, targets in pbar:
        # Move to device
        imgs = [img.to(device) for img in imgs]
        targets = targets_to_device(list(targets), device)

        optimizer.zero_grad()

        use_amp = (device.type == "cuda")
        with torch.autocast(device.type, enabled=use_amp):
            loss_dict = model(imgs, targets)
            total_loss = sum(loss_dict.values())

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            optimizer.step()

        # Accumulate
        lc = loss_dict.get("loss_classifier", torch.tensor(0.0))
        lb = loss_dict.get("loss_box_reg", torch.tensor(0.0))
        lm = loss_dict.get("loss_mask", torch.tensor(0.0))
        lo = loss_dict.get("loss_objectness", torch.tensor(0.0))
        lr_box = loss_dict.get(
            "loss_rpn_box_reg", torch.tensor(0.0)
        )

        sum_losses["loss_total"] += total_loss.item()
        sum_losses["loss_classifier"] += lc.item()
        sum_losses["loss_box_reg"] += lb.item()
        sum_losses["loss_mask"] += lm.item()
        sum_losses["loss_objectness"] += lo.item()
        sum_losses["loss_rpn_box_reg"] += lr_box.item()
        n_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{total_loss.item():.3f}",
                "cls": f"{lc.item():.2f}",
                "box": f"{lb.item():.2f}",
                "mask": f"{lm.item():.2f}",
                "rpn_obj": f"{lo.item():.2f}",
                "rpn_box": f"{lr_box.item():.2f}",
            },
            refresh=False,
        )

    # Compute means
    if n_batches == 0:
        return sum_losses
    return {k: v / n_batches for k, v in sum_losses.items()}


# ===========================================================================
# Validation / AP50 evaluation
# ===========================================================================


def evaluate_ap50(
    model,
    loader,
    device,
    coco_gt_dict: dict,
    epoch: int,
    total_epochs: int,
    score_threshold: float = 0.05,
) -> float:
    """Run inference on the val set and compute AP50.

    Parameters
    ----------
    model : nn.Module
        Mask R-CNN model (will be set to eval mode).
    loader : DataLoader
        Validation DataLoader.
    device : torch.device
    coco_gt_dict : dict
        Pre-built COCO ground-truth dict (from build_coco_gt).
    epoch : int
        Current epoch (1-based, for display only).
    total_epochs : int
    score_threshold : float
        Minimum score to include a prediction (default 0.05).

    Returns
    -------
    float
        AP @ IoU=0.50 (mask IoU).
    """
    model.eval()

    # paste_masks_in_image stacks N full-res masks as float32 on GPU.
    # With box_detections_per_img=800 on a ~1500px image that's ~7 GiB — OOM on 8GB cards.
    # Cap to 300 during val (safe: 300 * 1600^2 * 4B ≈ 3 GiB) and restore after.
    orig_det = model.roi_heads.detections_per_img
    model.roi_heads.detections_per_img = 300
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = []

    desc = f"Epoch {epoch:02d}/{total_epochs:02d} [Val  ]"
    pbar = tqdm(loader, desc=desc, leave=True)

    try:
        with torch.no_grad():
            for imgs, targets in pbar:
                imgs = [img.to(device) for img in imgs]
                targets_cpu = list(targets)

                # Get image_ids from targets (int64 tensor [1])
                img_ids = [
                    int(t["image_id"].item()) for t in targets_cpu
                ]

                preds = model(imgs)

                for pred, img_id in zip(preds, img_ids):
                    boxes = pred["boxes"]      # FloatTensor[N,4] xyxy
                    labels = pred["labels"]    # Int64Tensor[N]
                    scores = pred["scores"]    # FloatTensor[N]
                    masks = pred["masks"]      # FloatTensor[N,1,H,W]

                    if len(labels) == 0:
                        continue

                    # Filter by score threshold
                    keep = scores >= score_threshold
                    if not keep.any():
                        continue

                    boxes = boxes[keep].cpu()
                    labels = labels[keep].cpu()
                    scores = scores[keep].cpu()
                    masks = masks[keep].cpu()

                    # Binary masks: threshold at 0.5
                    bin_masks = (masks[:, 0] > 0.5).numpy()

                    for j in range(len(labels)):
                        x1, y1, x2, y2 = boxes[j].tolist()
                        w = x2 - x1
                        h = y2 - y1
                        rle = encode_mask(bin_masks[j])

                        results.append(
                            {
                                "image_id": img_id,
                                "category_id": int(labels[j].item()),
                                "bbox": [x1, y1, w, h],
                                "score": float(scores[j].item()),
                                "segmentation": rle,
                            }
                        )
    finally:
        model.roi_heads.detections_per_img = orig_det

    if len(results) == 0:
        return 0.0

    # Build COCO ground-truth object
    import io
    import contextlib

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dict
    # Suppress the default stdout print from createIndex
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.maxDets = [1, 10, 800]  # dataset can have >130 instances per image
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # stats[1] is AP @ IoU=0.50
    ap50 = float(coco_eval.stats[1])
    return ap50


# ===========================================================================
# Checkpoint helpers
# ===========================================================================


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scaler,
    scheduler,
    best_ap50: float,
):
    """Save a training checkpoint.

    Parameters
    ----------
    path : Path
    epoch : int
    model, optimizer, scaler, scheduler : training objects
    best_ap50 : float
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_ap50": best_ap50,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer, scaler, scheduler):
    """Load a training checkpoint, restoring all states in-place.

    Parameters
    ----------
    path : Path
    model, optimizer, scaler, scheduler : objects to restore

    Returns
    -------
    tuple
        (start_epoch, best_ap50)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_ap50 = float(ckpt.get("best_ap50", 0.0))
    print(
        f"Resumed from {path}  (epoch={ckpt['epoch']}, "
        f"best_ap50={best_ap50:.4f})"
    )
    return start_epoch, best_ap50


# ===========================================================================
# CSV log helper
# ===========================================================================

_CSV_HEADER = [
    "epoch",
    "loss_total",
    "loss_classifier",
    "loss_box_reg",
    "loss_mask",
    "loss_objectness",
    "loss_rpn_box_reg",
    "val_ap50",
    "lr",
    "elapsed_sec",
]


def append_csv_row(log_path: Path, row: dict):
    """Append one row to the CSV training log.

    Creates the file with a header if it does not exist yet.

    Parameters
    ----------
    log_path : Path
    row : dict
        Keys must match _CSV_HEADER.
    """
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _CSV_HEADER})


# ===========================================================================
# Argument parser
# ===========================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN on H&E cell instance data."
    )
    parser.add_argument(
        "--data-dir",
        default="hw3-data-release/train",
        help="Path to training data root (default: hw3-data-release/train)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Total training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Initial learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count (default: 4)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to train on (default: cuda if available)",
    )
    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- Directories -------------------------------------------------------
    ckpt_dir = Path("checkpoints")
    log_dir = Path("logs")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train_log.csv"

    # ---- Dataset -----------------------------------------------------------
    print(f"Loading dataset from: {args.data_dir}")
    full_dataset = CellDataset(args.data_dir, transforms=None)
    print(f"Total samples: {len(full_dataset)}")

    train_subset, val_subset = get_train_val_split(
        full_dataset, val_ratio=0.2, seed=42
    )
    train_aug_ds = CellDataset(args.data_dir, transforms=TrainTransform())
    train_subset = torch.utils.data.Subset(train_aug_ds, train_subset.indices)
    print(
        f"Train: {len(train_subset)}, Val: {len(val_subset)}"
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,  # bool masks can't be pinned safely
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    # ---- Build COCO GT once ------------------------------------------------
    print("Building COCO ground-truth for validation ...")
    coco_gt_dict = build_coco_gt(val_subset)
    print(
        f"  {len(coco_gt_dict['images'])} images, "
        f"{len(coco_gt_dict['annotations'])} annotations"
    )

    # ---- Model -------------------------------------------------------------
    model = build_model(num_classes=5)
    model.to(device)

    # ---- Optimizer & scheduler ---------------------------------------------
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1
    )
    scaler = torch.amp.GradScaler(
        args.device, enabled=(device.type == "cuda")
    )

    # ---- Resume ------------------------------------------------------------
    start_epoch = 1
    best_ap50 = 0.0

    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch, best_ap50 = load_checkpoint(
                resume_path, model, optimizer, scaler, scheduler
            )
        else:
            print(
                f"WARNING: --resume path not found: {resume_path}"
            )

    # ---- Training loop -----------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # -- Train -----------------------------------------------------------
        mean_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        # -- Val -------------------------------------------------------------
        ap50 = evaluate_ap50(
            model,
            val_loader,
            device,
            coco_gt_dict,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        elapsed = time.time() - t0

        # -- Report ----------------------------------------------------------
        improved = ap50 > best_ap50
        marker = " (*) -> saved best.pth" if improved else ""
        print(
            f"  AP50={ap50:.4f} | best={max(ap50, best_ap50):.4f}"
            f"{marker}"
        )

        if improved:
            best_ap50 = ap50
            save_checkpoint(
                ckpt_dir / "best.pth",
                epoch,
                model,
                optimizer,
                scaler,
                scheduler,
                best_ap50,
            )

        # -- Save last -------------------------------------------------------
        save_checkpoint(
            ckpt_dir / "last.pth",
            epoch,
            model,
            optimizer,
            scaler,
            scheduler,
            best_ap50,
        )

        # -- Save snapshot every 10 epochs -----------------------------------
        if epoch % 10 == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                model,
                optimizer,
                scaler,
                scheduler,
                best_ap50,
            )

        # -- CSV log ---------------------------------------------------------
        row = {
            "epoch": epoch,
            "loss_total": f"{mean_losses['loss_total']:.6f}",
            "loss_classifier": (
                f"{mean_losses['loss_classifier']:.6f}"
            ),
            "loss_box_reg": f"{mean_losses['loss_box_reg']:.6f}",
            "loss_mask": f"{mean_losses['loss_mask']:.6f}",
            "loss_objectness": (
                f"{mean_losses['loss_objectness']:.6f}"
            ),
            "loss_rpn_box_reg": (
                f"{mean_losses['loss_rpn_box_reg']:.6f}"
            ),
            "val_ap50": f"{ap50:.6f}",
            "lr": f"{current_lr:.8f}",
            "elapsed_sec": f"{elapsed:.1f}",
        }
        append_csv_row(log_path, row)

        print(
            f"  Epoch {epoch:02d} done | "
            f"loss={mean_losses['loss_total']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"time={elapsed:.1f}s"
        )

    print("\nTraining complete.")
    print(f"Best AP50: {best_ap50:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir.resolve()}")
    print(f"Log saved to: {log_path.resolve()}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # Smoke test hint:
    # python train.py --epochs 1 --batch-size 1 --num-workers 0
    main()
