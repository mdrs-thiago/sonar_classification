import os, math, json, random, shutil
from glob import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

from transformers import ViTForImageClassification, ViTImageProcessor, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------------------
# CONFIG
# ---------------------------
class CFG:
    data_dir = os.path.join("datasets","data_sonar")
    model_ckpt = "google/vit-base-patch16-224-in21k"  # ViT pré-treinado
    img_target = 224          # ViT padrão
    pad_value = 0             # valor de padding (depois normaliza)
    seed = 42

    # Treino
    epochs_bin = 10
    epochs_multi = 12
    batch_size = 32
    lr = 3e-5
    weight_decay = 0.05
    warmup_ratio = 0.05
    num_workers = 4
    amp = True  # mixed precision

    out_dir = "runs_vit_sonar"
    ckpt_bin = os.path.join(out_dir, "bin_best.pt")
    ckpt_multi = os.path.join(out_dir, "multi_best.pt")

    # split
    val_size = 0.15
    test_size = 0.15
    # se tiver grupos (ex.: por id de cena) você pode mapear aqui; mantemos None
    use_groups = False

# ---------------------------
# UTILIDADES
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_images_by_class(root: str) -> List[Tuple[str, str]]:
    """
    Retorna lista [(path, class_name)]
    Espera subpastas: object_a, object_b, object_c, no_object
    """
    items = []
    for cls in ["object_a", "object_b", "object_c", "no_object"]:
        cls_dir = os.path.join(root, cls)
        exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
        files = sum([glob(os.path.join(cls_dir, e)) for e in exts], [])
        for f in files:
            items.append((f, cls))
    return items

def build_label_maps() -> Tuple[Dict[str,int], Dict[int,str]]:
    # Multiclasse somente entre objetos (no_object fica fora)
    mc_map = {"object_a":0, "object_b":1, "object_c":2}
    mc_inv = {v:k for k,v in mc_map.items()}
    return mc_map, mc_inv

# ---------------------------
# TRANSFORMS com resize + pad e máscara de patches
# ---------------------------
class ResizePadForViT:
    """
    - Recebe imagens HxW (original: 256x50).
    - Redimensiona mantendo proporção para ter altura target (224) e largura proporcional (≈44).
    - Faz padding à DIREITA até largura target (224).
    - Retorna imagem PIL já 224x224.
    - Também calcula o número de colunas de patches VÁLIDAS (antes do padding).
    """
    def __init__(self, target_size: int = 224):
        self.target = target_size

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, int]:
        img = img.convert("RGB")
        w, h = img.size  # PIL: (W,H). Seus dados: 50x256? Se for 256x50, ajuste leitura. Assumimos W=50, H=256 normalmente.
        # Garantir que estamos tratando (H alto, W estreito). Se veio 256x50 já ok.
        # Redimensiona para altura target
        new_h = self.target
        new_w = int(round(w * (new_h / h)))
        if new_w > self.target:
            # fallback: garante que não ultrapasse target; reescala por largura
            new_w = self.target
            new_h = int(round(h * (new_w / w)))

        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
        # pad à direita
        pad_w = self.target - new_w
        if pad_w < 0: pad_w = 0
        new_img = Image.new("RGB", (self.target, self.target), (0,0,0))
        new_img.paste(img_resized, (0, 0))

        return new_img, new_w  # largura útil antes do padding

def make_attention_mask(new_w: int, target: int, patch: int) -> torch.Tensor:
    """
    Cria máscara de atenção no espaço de patches (flatten + CLS):
      - 1 para patches válidos (cobrem área não-paddada)
      - 0 para patches de padding (à direita)
    Para ViT-Base 224 e patch 16: grid 14x14. Se new_w≈44, colunas válidas ≈ ceil(44/16)=3.
    """
    grid = target // patch  # ex: 224//16=14
    valid_cols = math.ceil(new_w / patch)
    valid_cols = max(0, min(valid_cols, grid))
    mask = torch.zeros(grid, grid, dtype=torch.long)  # (rows=H, cols=W)
    if valid_cols > 0:
        mask[:, :valid_cols] = 1  # colunas à esquerda são válidas
    mask = mask.flatten()  # (grid*grid,)
    # prepend 1 para o token [CLS]
    mask = torch.cat([torch.ones(1, dtype=torch.long), mask], dim=0)  # seq_len = 1 + grid*grid
    return mask  # (seq_len,)

# ---------------------------
# DATASET
# ---------------------------
class SonarDataset(Dataset):
    def __init__(
        self,
        paths_labels: List[Tuple[str,str]],
        label_map: Dict[str,int],
        processor: ViTImageProcessor,
        target_size: int = 224,
        augment: bool = False,
        include_only_classes: Optional[List[str]] = None
    ):
        """
        paths_labels: lista (path, class_name)
        label_map: mapeia class_name -> label int
        include_only_classes: se fornecido, filtra exemplos
        """
        self.processor = processor
        self.target_size = target_size
        self.resize_pad = ResizePadForViT(target_size)
        self.patch = None  # será setado externamente com model.config.patch_size
        self.augment = augment
        self.label_map = label_map

        # filtra
        if include_only_classes is not None:
            paths_labels = [pl for pl in paths_labels if pl[1] in include_only_classes]
        self.samples = paths_labels

        # augmentações leves (opcionais)
        aug_list = []
        if augment:
            aug_list = [
                transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
                transforms.RandomAffine(degrees=3, translate=(0.02,0.02), scale=(0.98,1.02))
            ]
        self.aug = transforms.Compose(aug_list) if aug_list else None

    def set_patch(self, patch_size: int):
        self.patch = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cname = self.samples[idx]
        img = Image.open(path)
        # redimensiona + padding à direita
        img224, valid_w = self.resize_pad(img)
        if self.aug:
            img224 = self.aug(img224)

        # processor lida com to_tensor + normalização
        enc = self.processor(images=img224, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)  # (3,224,224)

        # construir attention_mask no espaço de patches
        if self.patch is None:
            raise RuntimeError("Patch size not set in dataset. Call dataset.set_patch(model.config.patch_size).")
        attn_mask = make_attention_mask(valid_w, target=self.target_size, patch=self.patch)  # (1+grid^2,)

        label = self.label_map[cname]
        return {
            "pixel_values": pixel_values,        # (3,224,224)
            "attention_mask": attn_mask,         # (seq_len,)
            "labels": torch.tensor(label).long(),
            "path": path,
            "class_name": cname
        }

# ---------------------------
# COLLATE
# ---------------------------
def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)  # (B,3,224,224)
    # atenção: seq_len é constante para um dado modelo (1 + 14*14 = 197), então podemos empilhar
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)  # (B,seq_len)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    paths = [b["path"] for b in batch]
    cnames = [b["class_name"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "labels": labels,
        "paths": paths,
        "class_names": cnames
    }

# ---------------------------
# LOOP DE TREINO
# ---------------------------
@dataclass
class TrainOut:
    best_acc: float
    best_path: str

def train_one_stage(
    train_ds: Dataset,
    val_ds: Dataset,
    num_labels: int,
    out_path: str,
    base_ckpt: Optional[str] = None,
    lr: float = 3e-5,
    epochs: int = 10,
    weight_decay: float = 0.05,
    warmup_ratio: float = 0.05,
    amp: bool = True
) -> TrainOut:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = train_ds.processor

    # Modelo
    if base_ckpt is None:
        model = ViTForImageClassification.from_pretrained(
            CFG.model_ckpt,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # útil se reposicionarmos cabeça depois
        )
    else:
        # Recarrega pesos de uma fase anterior (pode ter num_labels diferente)
        model = ViTForImageClassification.from_pretrained(
            base_ckpt if os.path.isdir(base_ckpt) else CFG.model_ckpt,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        # Se base_ckpt for caminho de .pt (state_dict), carregamos manualmente (exceto classifier)
        if base_ckpt.endswith(".pt") and os.path.isfile(base_ckpt):
            sd = torch.load(base_ckpt, map_location="cpu")
            # remover cabeça anterior se shapes divergirem
            miss = model.load_state_dict(sd, strict=False)
            # miss contem missing_keys/unexpected_keys — está ok para trocar a cabeça

    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, pin_memory=True, collate_fn=collate_fn)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = epochs * len(train_loader)
    num_warmup = int(warmup_ratio * num_training_steps)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup, num_training_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_acc = -1.0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        total, correct = 0, 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()

            preds = out.logits.argmax(dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = correct / total if total else 0.0

        # validação
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                out = model(pixel_values=pixel_values)
                preds = out.logits.argmax(dim=-1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro")

        print(f"[Epoch {epoch:02d}] train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"  ↳ novo melhor modelo salvo em {out_path}")

    return TrainOut(best_acc=best_acc, best_path=out_path)

def prepare_splits(all_items: List[Tuple[str,str]]):
    """
    Split estratificado em treino/val/test a partir das 4 classes.
    """
    paths = [p for p,_ in all_items]
    labels = [c for _,c in all_items]

    # primeiro separa test
    paths_trv, paths_te, labels_trv, labels_te = train_test_split(
        paths, labels, test_size=CFG.test_size, random_state=CFG.seed, stratify=labels
    )
    # depois separa val do restante
    val_ratio = CFG.val_size / (1.0 - CFG.test_size)
    paths_tr, paths_va, labels_tr, labels_va = train_test_split(
        paths_trv, labels_trv, test_size=val_ratio, random_state=CFG.seed, stratify=labels_trv
    )

    def pack(ps, ls):
        return list(zip(ps, ls))

    return pack(paths_tr, labels_tr), pack(paths_va, labels_va), pack(paths_te, labels_te)

# ---------------------------
# AVALIAÇÃO
# ---------------------------
def evaluate(model_path: str, ds: Dataset, label_names: List[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(
        CFG.model_ckpt, num_labels=len(label_names), ignore_mismatched_sizes=True
    )
    sd = torch.load(model_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False,
                        num_workers=CFG.num_workers, pin_memory=True, collate_fn=collate_fn)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            am = batch["attention_mask"].to(device)
            lab = batch["labels"].to(device)
            out = model(pixel_values=pv, attention_mask=am)
            pred = out.logits.argmax(dim=-1)
            y_true.extend(lab.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    print(f"ACC = {acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_names))

# ---------------------------
# MAIN
# ---------------------------
def main():
    set_seed(CFG.seed)
    os.makedirs(CFG.out_dir, exist_ok=True)

    # lista de (path, class_name)
    items = list_images_by_class(CFG.data_dir)
    assert len(items) > 0, f"Nenhuma imagem encontrada em {CFG.data_dir}"

    # split
    tr, va, te = prepare_splits(items)

    # Processador e datasets
    processor = ViTImageProcessor.from_pretrained(CFG.model_ckpt)
    mc_map, mc_inv = build_label_maps()

    # --------------- FASE 1: BINÁRIA ---------------
    # mapeamento binário: no_object -> 0, {object_a|b|c} -> 1
    def to_bin(items_pl):
        mapped = []
        for p,c in items_pl:
            mapped.append((p, "object" if c != "no_object" else "no_object"))
        return mapped

    bin_label_map = {"no_object": 0, "object": 1}
    # Datasets binários (mantém TODAS as imagens)
    ds_tr_bin = SonarDataset(to_bin(tr), bin_label_map, processor, target_size=CFG.img_target, augment=True)
    ds_va_bin = SonarDataset(to_bin(va), bin_label_map, processor, target_size=CFG.img_target, augment=False)
    ds_te_bin = SonarDataset(to_bin(te), bin_label_map, processor, target_size=CFG.img_target, augment=False)

    # setar patch size (do modelo)
    tmp_model = ViTForImageClassification.from_pretrained(CFG.model_ckpt)
    patch = tmp_model.config.patch_size
    for ds in [ds_tr_bin, ds_va_bin, ds_te_bin]:
        ds.set_patch(patch)

    print(">> Treinando fase binária (no_object vs object)...")
    out_bin = train_one_stage(
        ds_tr_bin, ds_va_bin, num_labels=2, out_path=CFG.ckpt_bin,
        base_ckpt=None, lr=CFG.lr, epochs=CFG.epochs_bin,
        weight_decay=CFG.weight_decay, warmup_ratio=CFG.warmup_ratio, amp=CFG.amp
    )
    print(f"Melhor ACC (val, binária): {out_bin.best_acc:.4f}")
    print(">> Avaliando no teste (binário)")
    evaluate(CFG.ckpt_bin, ds_te_bin, label_names=["no_object","object"])

    # --------------- FASE 2: MULTICLASSE ---------------
    # Agora treinamos apenas com objetos (exclui no_object) e reutilizamos os pesos
    include_objs = ["object_a", "object_b", "object_c"]
    ds_tr_mc = SonarDataset(tr, mc_map, processor, target_size=CFG.img_target, augment=True, include_only_classes=include_objs)
    ds_va_mc = SonarDataset(va, mc_map, processor, target_size=CFG.img_target, augment=False, include_only_classes=include_objs)
    ds_te_mc = SonarDataset(te, mc_map, processor, target_size=CFG.img_target, augment=False, include_only_classes=include_objs)
    for ds in [ds_tr_mc, ds_va_mc, ds_te_mc]:
        ds.set_patch(patch)

    print(">> Treinando fase multiclasses (a/b/c) reaproveitando pesos da binária...")
    out_mc = train_one_stage(
        ds_tr_mc, ds_va_mc, num_labels=3, out_path=CFG.ckpt_multi,
        base_ckpt=CFG.ckpt_bin, lr=CFG.lr, epochs=CFG.epochs_multi,
        weight_decay=CFG.weight_decay, warmup_ratio=CFG.warmup_ratio, amp=CFG.amp
    )
    print(f"Melhor ACC (val, multiclasses): {out_mc.best_acc:.4f}")
    print(">> Avaliando no teste (multiclasses)")
    evaluate(CFG.ckpt_multi, ds_te_mc, label_names=[mc_inv[i] for i in range(3)])

    # salva configs úteis
    meta = {
        "model_ckpt": CFG.model_ckpt,
        "patch_size": patch,
        "img_target": CFG.img_target,
        "bin_ckpt": CFG.ckpt_bin,
        "multi_ckpt": CFG.ckpt_multi,
    }
    with open(os.path.join(CFG.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(">> Finalizado.")

if __name__ == "__main__":
    main()

