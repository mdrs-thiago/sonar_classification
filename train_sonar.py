import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

try:
    from torchvision.models import ResNet18_Weights  # type: ignore[attr-defined]
except ImportError:  # for older torchvision versions
    ResNet18_Weights = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Sample:
    path: Path
    original_label: str


@dataclass(frozen=True)
class Entry:
    path: Path
    original_label: str
    target: int


class SonarImageDataset(Dataset):
    def __init__(self, entries: Sequence[Entry], transform: transforms.Compose):
        self.entries = list(entries)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image = Image.open(entry.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, entry.target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples(root: Path) -> List[Sample]:
    samples: List[Sample] = []
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for file_path in sorted(class_dir.glob("*")):
            if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                continue
            samples.append(Sample(path=file_path, original_label=class_dir.name))
    if not samples:
        raise RuntimeError(f"No images found under {root}")
    return samples


def split_indices(
    samples: Sequence[Sample],
    label_mapping: Dict[str, int],
    val_size: float,
    test_size: float,
    seed: int,
) -> Dict[str, List[int]]:
    indices = np.arange(len(samples))
    labels = np.array([label_mapping[s.original_label] for s in samples])

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    if val_size > 0:
        val_ratio = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_ratio,
            stratify=labels[train_idx],
            random_state=seed,
        )
    else:
        val_idx = np.array([], dtype=int)

    return {
        "train": sorted(train_idx.tolist()),
        "val": sorted(val_idx.tolist()),
        "test": sorted(test_idx.tolist()),
    }


def build_entries(
    samples: Sequence[Sample],
    indices: Sequence[int],
    mapping: Dict[str, int],
    include: Optional[Sequence[str]] = None,
) -> List[Entry]:
    include_set = set(include) if include is not None else None
    entries: List[Entry] = []
    for idx in indices:
        sample = samples[idx]
        if include_set is not None and sample.original_label not in include_set:
            continue
        if sample.original_label not in mapping:
            continue
        entries.append(Entry(path=sample.path, original_label=sample.original_label, target=mapping[sample.original_label]))
    return entries


def make_dataloaders(
    split_entries: Dict[str, List[Entry]],
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    seed: int,
    train_sampler: Optional[WeightedRandomSampler] = None,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset = SonarImageDataset(split_entries["train"], train_transform)
    val_dataset = SonarImageDataset(split_entries["val"], eval_transform) if split_entries["val"] else None
    test_dataset = SonarImageDataset(split_entries["test"], eval_transform)

    if train_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            generator=generator,
        )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(pretrained=True)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    device: torch.device,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    model.to(device)
    weight_tensor = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history: List[Dict[str, float]] = []
    best_state = None
    best_metric = -float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, dim=1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == targets).item()
            total += images.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = running_corrects / max(total, 1)

        val_loss = 0.0
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            correct = 0
            samples_count = 0
            loss_sum = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    _, preds = torch.max(outputs, dim=1)
                    loss_sum += loss.item() * images.size(0)
                    correct += torch.sum(preds == targets).item()
                    samples_count += images.size(0)
            val_loss = loss_sum / max(samples_count, 1)
            val_acc = correct / max(samples_count, 1)
            metric = val_acc
        else:
            metric = epoch_acc

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def inference(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    return all_targets, all_preds


def print_class_distribution(title: str, entries: Sequence[Entry]) -> None:
    counter = Counter(e.target for e in entries)
    total = len(entries)
    print(title)
    for label, count in sorted(counter.items()):
        share = (count / total) * 100 if total else 0.0
        print(f"  label {label}: {count} samples ({share:.1f}%)")
    print("")


def prepare_balancing(
    entries: Sequence[Entry],
    num_classes: int,
    seed: int,
    strategy: str,
) -> Tuple[Optional[WeightedRandomSampler], Optional[torch.Tensor]]:
    """Return sampler and per-class weights according to chosen strategy."""
    if not entries or strategy == "none":
        return None, None

    counts = Counter(e.target for e in entries)
    total = len(entries)
    class_weights = torch.ones(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        count = counts.get(cls, 0)
        if count > 0:
            class_weights[cls] = total / (num_classes * float(count))
        else:
            class_weights[cls] = 0.0

    mean_weight = class_weights[class_weights > 0].mean() if torch.any(class_weights > 0) else None
    if mean_weight is not None and mean_weight.item() > 0:
        class_weights = class_weights / mean_weight

    sampler: Optional[WeightedRandomSampler] = None
    if strategy in {"sampler", "both"}:
        sample_weights = [class_weights[e.target].item() for e in entries]
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)

    class_weight_tensor: Optional[torch.Tensor] = None
    if strategy in {"class-weight", "both"}:
        class_weight_tensor = class_weights

    return sampler, class_weight_tensor


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def evaluate_and_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, dict]:
    y_true, y_pred = inference(model, loader, device)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix:")
    print(matrix)
    return {"classification_report": report, "confusion_matrix": matrix.tolist()}


def hierarchical_pipeline(
    args,
    samples: Sequence[Sample],
    split_idx: Dict[str, List[int]],
    device: torch.device,
) -> Dict[str, dict]:
    object_presence_map = {
        "no_object": 0,
        "object_a": 1,
        "object_b": 1,
        "object_c": 1,
    }
    interest_map = {
        "object_a": 1,
        "object_b": 1,
        "object_c": 0,
    }
    interest_classes = {"object_a", "object_b", "object_c"}

    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transforms_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    stage1_entries = {split: build_entries(samples, indices, object_presence_map) for split, indices in split_idx.items()}
    print_class_distribution("Stage 1 class distribution", stage1_entries["train"])
    stage1_sampler, stage1_class_weights = prepare_balancing(
        stage1_entries["train"],
        num_classes=2,
        seed=args.seed,
        strategy=args.balance_strategy,
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        stage1_entries,
        transforms_train,
        transforms_eval,
        args.batch_size,
        args.num_workers,
        args.seed,
        train_sampler=stage1_sampler,
    )
    stage1_model = build_model(num_classes=2, freeze_backbone=args.freeze_backbone)
    stage1_model, stage1_history = train_one_model(
        stage1_model,
        train_loader,
        val_loader,
        args.epochs,
        device,
        args.lr,
        args.weight_decay,
        args.use_amp and device.type == "cuda",
        class_weights=stage1_class_weights,
    )
    print("\nStage 1 evaluation:")
    stage1_metrics = evaluate_and_report(stage1_model, test_loader, device, class_names=["no_object", "object_present"])

    stage2_entries = {
        split: build_entries(samples, indices, interest_map, include=interest_classes)
        for split, indices in split_idx.items()
    }
    if not stage2_entries["train"] or not stage2_entries["test"]:
        raise RuntimeError("Stage 2 split has no samples. Check dataset balance.")
    print_class_distribution("Stage 2 class distribution", stage2_entries["train"])
    stage2_sampler, stage2_class_weights = prepare_balancing(
        stage2_entries["train"],
        num_classes=2,
        seed=args.seed,
        strategy=args.balance_strategy,
    )
    train_loader2, val_loader2, test_loader2 = make_dataloaders(
        stage2_entries,
        transforms_train,
        transforms_eval,
        args.batch_size,
        args.num_workers,
        args.seed,
        train_sampler=stage2_sampler,
    )
    stage2_model = build_model(num_classes=2, freeze_backbone=args.freeze_backbone)
    stage2_model, stage2_history = train_one_model(
        stage2_model,
        train_loader2,
        val_loader2,
        args.epochs,
        device,
        args.lr,
        args.weight_decay,
        args.use_amp and device.type == "cuda",
        class_weights=stage2_class_weights,
    )
    print("\nStage 2 evaluation:")
    stage2_metrics = evaluate_and_report(stage2_model, test_loader2, device, class_names=["another_object", "object_of_interest"])

    if args.output_dir:
        ensure_output_dir(Path(args.output_dir))
        torch.save(stage1_model.state_dict(), Path(args.output_dir) / "hierarchical_stage1.pt")
        torch.save(stage2_model.state_dict(), Path(args.output_dir) / "hierarchical_stage2.pt")
        save_json({"history": stage1_history}, Path(args.output_dir) / "hierarchical_stage1_history.json")
        save_json({"history": stage2_history}, Path(args.output_dir) / "hierarchical_stage2_history.json")

    direct_mapping = {
        "no_object": 0,
        "object_a": 1,
        "object_b": 1,
        "object_c": 2,
    }
    final_class_names = ["no_object", "object_of_interest", "another_object"]

    stage1_model.eval()
    stage2_model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for entry in build_entries(samples, split_idx["test"], direct_mapping):
            image = Image.open(entry.path).convert("RGB")
            image = transforms_eval(image).unsqueeze(0).to(device)

            stage1_out = stage1_model(image)
            stage1_pred = torch.argmax(stage1_out, dim=1).item()
            if stage1_pred == 0:
                final_pred = 0
            else:
                stage2_out = stage2_model(image)
                stage2_pred = torch.argmax(stage2_out, dim=1).item()
                final_pred = 1 if stage2_pred == 1 else 2
            y_pred.append(final_pred)
            y_true.append(direct_mapping[entry.original_label])

    print("\nHierarchical pipeline (combined) evaluation:")
    print(classification_report(y_true, y_pred, target_names=final_class_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("Confusion matrix:")
    print(cm)

    combined_metrics = {
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=final_class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
    }

    results = {
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "combined": combined_metrics,
    }

    if args.output_dir:
        save_json(results, Path(args.output_dir) / "hierarchical_metrics.json")

    return results


def direct_pipeline(
    args,
    samples: Sequence[Sample],
    split_idx: Dict[str, List[int]],
    device: torch.device,
) -> Dict[str, dict]:
    direct_mapping = {
        "no_object": 0,
        "object_a": 1,
        "object_b": 1,
        "object_c": 2,
    }
    class_names = ["no_object", "object_of_interest", "another_object"]

    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transforms_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    entries = {split: build_entries(samples, indices, direct_mapping) for split, indices in split_idx.items()}
    print_class_distribution("Direct pipeline class distribution", entries["train"])
    direct_sampler, direct_class_weights = prepare_balancing(
        entries["train"],
        num_classes=3,
        seed=args.seed,
        strategy=args.balance_strategy,
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        entries,
        transforms_train,
        transforms_eval,
        args.batch_size,
        args.num_workers,
        args.seed,
        train_sampler=direct_sampler,
    )
    model = build_model(num_classes=3, freeze_backbone=args.freeze_backbone)
    model, history = train_one_model(
        model,
        train_loader,
        val_loader,
        args.epochs,
        device,
        args.lr,
        args.weight_decay,
        args.use_amp and device.type == "cuda",
        class_weights=direct_class_weights,
    )
    print("\nDirect pipeline evaluation:")
    metrics = evaluate_and_report(model, test_loader, device, class_names=class_names)

    if args.output_dir:
        ensure_output_dir(Path(args.output_dir))
        torch.save(model.state_dict(), Path(args.output_dir) / "direct_model.pt")
        save_json({"history": history}, Path(args.output_dir) / "direct_history.json")
        save_json(metrics, Path(args.output_dir) / "direct_metrics.json")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sonar classifiers using hierarchical and direct approaches.")
    parser.add_argument("--data-dir", type=str, default="datasets/data_sonar", help="Path to dataset root.")
    parser.add_argument("--output-dir", type=str, default="runs_sonar", help="Directory to store weights and logs.")
    parser.add_argument("--approach", type=str, choices=["hierarchical", "direct", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--balance-strategy",
        type=str,
        choices=["none", "class-weight", "sampler", "both"],
        default="both",
        help="Handle class imbalance via loss re-weighting, sampling, or both.",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision training when possible.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze the feature extractor.")
    parser.add_argument("--device", type=str, default="auto", help="Force device (cpu or cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (
        torch.device(args.device)
        if args.device != "auto"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    samples = collect_samples(data_dir)

    direct_mapping = {
        "no_object": 0,
        "object_a": 1,
        "object_b": 1,
        "object_c": 2,
    }
    split_idx = split_indices(samples, direct_mapping, args.val_size, args.test_size, args.seed)

    results: Dict[str, dict] = {}
    if args.approach in {"hierarchical", "both"}:
        print("\n=== Training hierarchical pipeline ===")
        results["hierarchical"] = hierarchical_pipeline(args, samples, split_idx, device)
    if args.approach in {"direct", "both"}:
        print("\n=== Training direct pipeline ===")
        results["direct"] = direct_pipeline(args, samples, split_idx, device)

    if args.output_dir:
        ensure_output_dir(Path(args.output_dir))
        save_json(results, Path(args.output_dir) / "summary.json")


if __name__ == "__main__":
    main()
