# train_dogs_bbox.py

import os
import json
import time
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm

# ==============================
# 1. КОНФИГ
# ==============================

# Путь к папке Images из датасета Stanford Dogs (Kaggle)
# Пример: Path("/Users/you/data/stanford-dogs/Images")
DATA_DIR_IMAGES = Path("images/Images")

OUTPUT_DIR = Path("outputs_dogs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15
VAL_SPLIT = 0.1
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SEED = 42

# Использовать ли bounding boxes из аннотаций (если найдутся)
USE_BBOX_FROM_ANNOTATIONS = True

# Фиксируем сиды
random.seed(SEED)
torch.manual_seed(SEED)

# Определяем устройство
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


# ==============================
# 2. ТРАНСФОРМЫ
# ==============================

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ==============================
# 3. ПОИСК КАТАЛОГА АННОТАЦИЙ
# ==============================

def auto_find_annotation_root(images_dir: Path) -> Optional[Path]:
    """
    Ищем корень аннотаций.
    Твоя структура:
      <root>/Images
      <root>/annotations/Annotation
    """
    root = images_dir.parent  # <root>

    candidates = [
        root / "annotations" / "Annotation",   # <-- твой случай
        root / "Annotation",                   # иногда так
        root / "annotations",                  # на всякий
        root.parent / "annotations" / "Annotation",
        root.parent / "Annotation",
    ]

    for c in candidates:
        if c.is_dir():
            return c
    return None


# ==============================
# 4. CASTOM DATASET C BBOX
# ==============================

class DogsBBoxDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform,
        annotation_root: Optional[Path] = None,
        use_bbox: bool = True,
    ):
        """
        samples: список (путь_к_картинке, label) как в ImageFolder.samples
        transform: torchvision.transforms
        annotation_root: корневая папка Annotation/..., если есть
        use_bbox: использовать ли bbox, если файл аннотации найден
        """
        self.samples = samples
        self.transform = transform
        self.annotation_root = annotation_root
        self.use_bbox = use_bbox and (annotation_root is not None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        img_path = Path(path)
        img = Image.open(img_path).convert("RGB")

        if self.use_bbox:
            img = self._apply_bbox_if_exists(img, img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def _apply_bbox_if_exists(self, img: Image.Image, img_path: Path) -> Image.Image:
        """
        Пытаемся найти файл аннотации и вырезать собаку по bbox.
        Если что-то идёт не так — возвращаем оригинальное изображение.
        """
        try:
            if self.annotation_root is None:
                return img

            breed_folder = img_path.parent.name
            stem = img_path.stem  # без расширения

            candidates = [
                self.annotation_root / breed_folder / stem,
                self.annotation_root / breed_folder / f"{stem}.xml",
            ]

            ann_path = None
            for c in candidates:
                if c.exists():
                    ann_path = c
                    break

            if ann_path is None:
                return img

            tree = ET.parse(str(ann_path))
            root = tree.getroot()
            bnd = root.find(".//bndbox")
            if bnd is None:
                return img

            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))

            w, h = img.size
            xmin = max(0, min(xmin, w - 1))
            xmax = max(0, min(xmax, w))
            ymin = max(0, min(ymin, h - 1))
            ymax = max(0, min(ymax, h))

            if xmax <= xmin or ymax <= ymin:
                return img

            return img.crop((xmin, ymin, xmax, ymax))

        except Exception:
            # Если аннотация битая или формат другой — просто не режем
            return img


# ==============================
# 5. ДАТАЛОАДЕРЫ
# ==============================

def build_dataloaders():
    if not DATA_DIR_IMAGES.exists():
        raise FileNotFoundError(f"DATA_DIR_IMAGES не существует: {DATA_DIR_IMAGES}")

    # базовый ImageFolder только чтобы получить список файлов и классы
    base_ds = datasets.ImageFolder(DATA_DIR_IMAGES)
    samples = base_ds.samples  # список (path, label)
    class_names = base_ds.classes
    num_classes = len(class_names)
    num_samples = len(samples)

    print(f"Найдено изображений: {num_samples}, классов (пород): {num_classes}")

    annotation_root = None
    if USE_BBOX_FROM_ANNOTATIONS:
        annotation_root = auto_find_annotation_root(DATA_DIR_IMAGES)
        if annotation_root is not None:
            print(f"Каталог аннотаций найден: {annotation_root}")
            print("Будем обрезать изображения по bounding box, когда это возможно.")
        else:
            print("Каталог аннотаций не найден. Используем полные изображения.")
    else:
        print("USE_BBOX_FROM_ANNOTATIONS = False, bbox использоваться не будут.")

    # случайно перемешиваем индексы и делим на train/val
    indices = list(range(num_samples))
    random.shuffle(indices)

    val_size = int(num_samples * VAL_SPLIT)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    train_ds = DogsBBoxDataset(
        train_samples,
        transform=train_tfms,
        annotation_root=annotation_root,
        use_bbox=USE_BBOX_FROM_ANNOTATIONS,
    )
    val_ds = DogsBBoxDataset(
        val_samples,
        transform=test_tfms,
        annotation_root=annotation_root,
        use_bbox=USE_BBOX_FROM_ANNOTATIONS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE != "cpu"),
    )

    return train_loader, val_loader, class_names, num_classes, len(train_samples), len(val_samples)


# ==============================
# 6. МОДЕЛЬ (ResNet50 + веса)
# ==============================

def build_model(num_classes: int) -> nn.Module:
    # Предобученные веса ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Сначала замораживаем все слои
    for p in model.parameters():
        p.requires_grad = False

    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, num_classes),
    )
    return model


# ==============================
# 7. ОБУЧЕНИЕ / ВАЛИДАЦИЯ
# ==============================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Val", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def fit():
    train_loader, val_loader, class_names, num_classes, n_train, n_val = build_dataloaders()
    num_samples_total = n_train + n_val

    model = build_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0
    history = []

    print("\n=== Начинаем обучение ===")
    print(f"Эпох: {EPOCHS}, batch_size: {BATCH_SIZE}, train={n_train}, val={n_val}\n")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

        epoch_time = time.time() - epoch_start
        imgs_per_sec = num_samples_total / epoch_time if epoch_time > 0 else 0.0

        # После 3-й эпохи размораживаем backbone и уменьшаем LR
        if epoch == 3:
            print("Размораживаем все слои и уменьшаем learning rate...")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR / 10, weight_decay=WEIGHT_DECAY)

        print(
            f"[Эпоха {epoch:02d}/{EPOCHS}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}% | "
            f"time={epoch_time:.1f}s  ({imgs_per_sec:.1f} img/s)"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_sec": epoch_time,
                "imgs_per_sec": imgs_per_sec,
            }
        )

        # сохраняем лучший чекпоинт
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            with open(OUTPUT_DIR / "classes.json", "w", encoding="utf-8") as f:
                json.dump(class_names, f, ensure_ascii=False, indent=2)
            print(f"--> Новый лучший чекпоинт сохранён, val_acc={best_val_acc*100:.2f}%")

    with open(OUTPUT_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\n=== Обучение завершено ===")
    print(f"Лучшая val_acc: {best_val_acc*100:.2f}%")
    print(f"Файлы сохранены в: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    fit()
