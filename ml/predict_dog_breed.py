import os
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Те же настройки, что и в train-скрипте
IMG_SIZE = 224
OUTPUT_DIR = Path("outputs_dogs")

# Определяем устройство
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Трансформ для инференса (как val/test в обучении)
test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False

    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, num_classes),
    )
    return model


def load_for_inference(
    model_path: Path = OUTPUT_DIR / "best_model.pt",
    classes_path: Path = OUTPUT_DIR / "classes.json",
):
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден файл модели: {model_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Не найден файл классов: {classes_path}")

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names: List[str] = json.load(f)

    num_classes = len(class_names)
    model = build_model(num_classes)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, class_names


def prettify_breed(class_name: str) -> str:
    if "-" in class_name:
        class_name = class_name.split("-", 1)[1]
    class_name = class_name.replace("_", " ")
    return class_name

@torch.no_grad()
def predict_image(model, class_names, img_path, topk=5):
    img = Image.open(img_path).convert("RGB")
    x = test_tfms(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    vals, idxs = probs.topk(topk)

    results = []
    for v, i in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
        breed_name = prettify_breed(class_names[i])  # <-- вот тут
        results.append((breed_name, v))
    return results


def main():
    model, class_names = load_for_inference()
    print("Модель загружена. Можно делать предсказания.")

    while True:
        path = input("\nУкажи путь к изображению собаки (Enter, чтобы выйти): ").strip()
        if not path:
            print("Выход.")
            break

        if not os.path.exists(path):
            print("Файл не найден, попробуй ещё раз.")
            continue

        try:
            results = predict_image(model, class_names, path, topk=5)
        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")
            continue

        print(f"\nПредсказание для {path}:")
        for breed, prob in results:
            print(f"  {breed:40s}  {prob*100:6.2f}%")

        best_breed, best_prob = results[0]
        print(f"\n--> Итог: скорее всего это порода: **{best_breed}** ({best_prob*100:.2f}%)")


if __name__ == "__main__":
    main()
