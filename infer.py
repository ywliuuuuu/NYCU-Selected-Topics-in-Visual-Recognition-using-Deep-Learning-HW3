"""Inference script for Instance Segmentation (Mask R-CNN) on H&E cell images."""

import sys
import json
import zipfile
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset
from train import build_model, encode_mask

if sys.stdout.encoding != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Mask R-CNN inference for instance segmentation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--test-dir", type=str, default="hw3-data-release/test_release")
    parser.add_argument(
        "--json-path",
        type=str,
        default="hw3-data-release/test_image_name_to_ids.json",
    )
    parser.add_argument("--output", type=str, default="test-results.json")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load Mask R-CNN from a training checkpoint."""
    model = build_model(num_classes=5)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    best_ap50 = ckpt.get("best_ap50", None)
    epoch = ckpt.get("epoch", None)
    print(f"Loaded checkpoint from epoch {epoch}, best AP50={best_ap50}")
    return model


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float,
) -> list:
    """Run model inference over all test images and return COCO-style results."""
    results = []

    with torch.no_grad():
        for image_tensor, image_id in tqdm(dataloader, desc="Inference"):
            image = image_tensor[0].to(device)
            img_id = int(image_id[0])

            preds = model([image])
            pred = preds[0]

            boxes = pred["boxes"].cpu()
            labels = pred["labels"].cpu()
            scores = pred["scores"].cpu()
            masks = pred["masks"].cpu()

            keep = scores >= score_threshold
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            masks = masks[keep]

            binary_masks = (masks[:, 0] > 0.5).numpy()

            for i in range(len(scores)):
                x1, y1, x2, y2 = boxes[i].tolist()
                bbox = [x1, y1, x2 - x1, y2 - y1]
                rle = encode_mask(binary_masks[i])
                results.append(
                    {
                        "image_id": img_id,
                        "bbox": bbox,
                        "score": float(scores[i]),
                        "category_id": int(labels[i]),
                        "segmentation": rle,
                    }
                )

    return results


def save_results(results: list, output_path: str) -> None:
    """Save COCO-style predictions to JSON and zip for CodaBench submission."""
    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(results, fh)

    zip_path = Path("submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_file, "test-results.json")

    print(f"Total predictions : {len(results)}")
    print(f"Results saved to  : {output_file.resolve()}")
    print(f"Submission zip    : {zip_path.resolve()}")


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)

    dataset = TestDataset(
        test_dir=args.test_dir,
        json_path=args.json_path,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    results = run_inference(model, dataloader, device, args.score_threshold)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
