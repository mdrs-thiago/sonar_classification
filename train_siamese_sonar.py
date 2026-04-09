import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from torchvision.models import ResNet18_Weights  # type: ignore[attr-defined]
except ImportError:  # fallback for older torchvision
    ResNet18_Weights = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Sample:
    path: Path
    original_label: str
    binary_label: int


@dataclass(frozen=True)
class Entry:
    path: Path
    target: int


class SiamesePairDataset(Dataset):
    def __init__(
        self,
        entries: Sequence[Entry],
        transform: transforms.Compose,
        positive_ratio: float = 0.5,
    ):
        self.entries = list(entries)
        self.transform = transform
        self.positive_ratio = positive_ratio

        groups: Dict[int, List[Entry]] = defaultdict(list)
        for entry in self.entries:
            groups[entry.target].append(entry)
        self.groups = groups
        self.labels = sorted(groups.keys())

        self.valid_positive_labels = [lbl for lbl, items in groups.items() if len(items) > 1]

        if not self.groups or any(len(items) == 0 for items in self.groups.values()):
            raise RuntimeError("Siamese dataset needs at least one sample per class.")
        if len(self.groups) < 2:
            raise RuntimeError("Siamese dataset needs at least two classes for negative pairs.")

    def __len__(self) -> int:
        return len(self.entries)

    def _sample_positive(self) -> Tuple[Entry, Entry, int]:
        if not self.valid_positive_labels:
            raise RuntimeError("No class has enough items for positive pairs.")
        label = random.choice(self.valid_positive_labels)
        candidates = self.groups[label]
        first, second = random.sample(candidates, 2)
        return first, second, 1

    def _sample_negative(self) -> Tuple[Entry, Entry, int]:
        label_a, label_b = random.sample(self.labels, 2)
        entry_a = random.choice(self.groups[label_a])
        entry_b = random.choice(self.groups[label_b])
        return entry_a, entry_b, 0

    def __getitem__(self, idx: int):
        make_positive = random.random() < self.positive_ratio and self.valid_positive_labels
        if make_positive:
            entry_a, entry_b, target = self._sample_positive()
        else:
            entry_a, entry_b, target = self._sample_negative()

        img_a = Image.open(entry_a.path).convert("RGB")
        img_b = Image.open(entry_b.path).convert("RGB")
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b, torch.tensor(target, dtype=torch.float32)


class SonarSingleDataset(Dataset):
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


class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim: int, freeze_backbone: bool):
        super().__init__()
        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
        else:
            backbone = models.resnet18(pretrained=True)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        in_features = backbone.fc.in_features
        self.head = nn.Linear(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        embeddings = self.head(features)
        return F.normalize(embeddings, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        distances = F.pairwise_distance(output1, output2)
        positive_loss = target * (distances ** 2)
        negative_loss = (1 - target) * torch.clamp(self.margin - distances, min=0.0) ** 2
        return torch.mean(positive_loss + negative_loss)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_samples(root: Path) -> List[Sample]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    mapping = {
        "object_a": 1,
        "object_b": 1,
        "object_c": 0,
        "no_object": 0,
    }
    samples: List[Sample] = []
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        label_name = class_dir.name
        if label_name not in mapping:
            continue
        binary_label = mapping[label_name]
        for file_path in sorted(class_dir.iterdir()):
            if file_path.suffix.lower() not in exts:
                continue
            samples.append(Sample(path=file_path, original_label=label_name, binary_label=binary_label))
    if not samples:
        raise RuntimeError(f"No usable images found in {root}")
    return samples


def split_indices(samples: Sequence[Sample], val_size: float, test_size: float, seed: int) -> Dict[str, List[int]]:
    indices = np.arange(len(samples))
    labels = np.array([s.binary_label for s in samples])

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


def build_entries(samples: Sequence[Sample], indices: Sequence[int]) -> List[Entry]:
    return [Entry(path=samples[i].path, target=samples[i].binary_label) for i in indices]


def make_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform


def build_prototypes(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[int, torch.Tensor]:
    model.eval()
    features: Dict[int, List[torch.Tensor]] = defaultdict(list)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            embeddings = model(images)
            for emb, tgt in zip(embeddings, targets):
                features[int(tgt.item())].append(emb.cpu())
    prototypes: Dict[int, torch.Tensor] = {}
    for label, vecs in features.items():
        if vecs:
            stacked = torch.stack(vecs, dim=0)
            prototypes[label] = F.normalize(stacked.mean(dim=0), p=2, dim=0)
    return prototypes


def evaluate_embeddings(
    model: nn.Module,
    loader: DataLoader,
    prototypes: Dict[int, torch.Tensor],
    device: torch.device,
    class_names: Sequence[str],
) -> Dict[str, object]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            embeddings = model(images)
            embeddings = embeddings.cpu()
            for emb, tgt in zip(embeddings, targets):
                scores = {}
                for label, proto in prototypes.items():
                    scores[label] = F.cosine_similarity(emb.unsqueeze(0), proto.unsqueeze(0)).item()
                pred = max(scores.items(), key=lambda kv: kv[1])[0]
                y_true.append(int(tgt.item()))
                y_pred.append(pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    return {"classification_report": report}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    use_amp: bool,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for img_a, img_b, targets in train_loader:
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                emb_a = model(img_a)
                emb_b = model(img_b)
                loss = criterion(emb_a, emb_b, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_count += 1

        epoch_loss = running_loss / max(batch_count, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"Epoch {epoch}: loss={epoch_loss:.4f}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Siamese network for sonar object-of-interest detection.")
    parser.add_argument("--data-dir", type=str, default="datasets/data_sonar")
    parser.add_argument("--output-dir", type=str, default="runs_siamese")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
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
    split_idx = split_indices(samples, args.val_size, args.test_size, args.seed)

    train_entries = build_entries(samples, split_idx["train"])
    val_entries = build_entries(samples, split_idx["val"])
    test_entries = build_entries(samples, split_idx["test"])

    train_transform, eval_transform = make_transforms(args.image_size)

    pair_train_dataset = SiamesePairDataset(train_entries, train_transform, positive_ratio=args.positive_ratio)
    pair_train_loader = DataLoader(
        pair_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    single_train_dataset = SonarSingleDataset(train_entries, eval_transform)
    single_val_dataset = SonarSingleDataset(val_entries, eval_transform) if val_entries else None
    single_test_dataset = SonarSingleDataset(test_entries, eval_transform)

    single_train_loader = DataLoader(
        single_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    single_val_loader = (
        DataLoader(
            single_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        if single_val_dataset is not None
        else None
    )
    single_test_loader = DataLoader(
        single_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = EmbeddingNet(args.embedding_dim, freeze_backbone=args.freeze_backbone)
    criterion = ContrastiveLoss(args.margin)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    history = train(
        model,
        pair_train_loader,
        optimizer,
        criterion,
        device,
        args.epochs,
        args.use_amp and device.type == "cuda",
    )

    prototypes = build_prototypes(model, single_train_loader, device)
    class_names = ["other", "object_of_interest"]
    val_metrics: Optional[Dict[str, object]] = None
    if single_val_loader is not None:
        val_metrics = evaluate_embeddings(model, single_val_loader, prototypes, device, class_names)
        print("\nValidation metrics:")
        print(json.dumps(val_metrics, indent=2))

    test_metrics = evaluate_embeddings(model, single_test_loader, prototypes, device, class_names)
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "prototypes": prototypes}, output_dir / "siamese_model.pt")
    results = {
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with (output_dir / "siamese_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
