# 1. 기본 설정
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import GroupKFold
import time
import copy
from tqdm import tqdm

DATA_DIR = "/content"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
DRIVER_LIST_FILE = os.path.join(DATA_DIR, "driver_imgs_list.csv")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 10
LR = 1e-4
N_FOLDS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# 2. 데이터 전처리
## Dataset 정의
class StateFarmDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["classname"], row["img"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["classname"][1:])
        return image, label

## Transform
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ]),
    "val": transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# 3. 훈련
def train_one_fold(fold_idx, train_df, val_df):
    print(f"\n--- Fold {fold_idx+1}/{N_FOLDS} Start ---")

    train_ds = StateFarmDataset(train_df, TRAIN_IMG_DIR, transform=data_transforms["train"])
    val_ds = StateFarmDataset(val_df, TRAIN_IMG_DIR, transform=data_transforms["val"])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    # VGGNet 모델 정의
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    class SimpleHead(nn.Module):
        def __init__(self, in_features, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    model.classifier = SimpleHead(512, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        ## TRAIN
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for inputs, labels in tqdm(train_loader, desc="Train", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

        train_loss /= len(train_ds)
        train_acc = train_corrects.double() / len(train_ds)

        ## VALIDATION
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Valid", leave=False):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_ds)
        val_acc = val_corrects.double() / len(val_ds)

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f"vgg19_mark26_fold{fold_idx}.pth")
            print(f"Best Model Saved! (Val Loss: {best_loss:.4f})")

    print(f"Fold {fold_idx+1} Finished. Best Val Loss: {best_loss:.4f}")
    return best_loss


df = pd.read_csv(DRIVER_LIST_FILE)

gkf = GroupKFold(n_splits=N_FOLDS)
splits = list(gkf.split(df, df["classname"], df["subject"]))

fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(splits):
    train_fold_df = df.iloc[train_idx].copy()
    val_fold_df = df.iloc[val_idx].copy()

    val_loss = train_one_fold(fold_idx, train_fold_df, val_fold_df)
    fold_results.append(val_loss)

print("\n--- All Folds Finished ---")
print(f"Average Validation Loss: {np.mean(fold_results):.4f}")


# 4. 테스트
TEST_DIR = os.path.join("/content", "test")
SAMPLE_SUB = "/content/sample_submission.csv"
sub_df = pd.read_csv(SAMPLE_SUB)
img_names = sub_df["img"].tolist()

transform_centercrop = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_fivecrop = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.FiveCrop(IMG_SIZE),
    transforms.Lambda(
        lambda crops: torch.stack(
            [
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                )(transforms.ToTensor()(crop))
                for crop in crops
            ]
        )
    ),
])

transform_resize = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

models_list = []
print("Loading models...")
for i in range(N_FOLDS):
    model = models.vgg19(weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    class SimpleHead(nn.Module):
        def __init__(self, in_features, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    model.classifier = SimpleHead(512, NUM_CLASSES)
    model = model.to(DEVICE)

    path = f"vgg19_mark26_fold{i}.pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models_list.append(model)
    else:
        print(f"Warning: {path} not found.")

print(f"Loaded {len(models_list)} models for ensemble.")


# 5. 추론 (Batch + TTA + Ensemble)
final_preds = []
with torch.no_grad():
    for i in tqdm(range(0, len(img_names), BATCH_SIZE), desc="Predicting with TTA"):
        batch_files = img_names[i : i + BATCH_SIZE]
        current_batch_size = len(batch_files)

        pil_images = []
        for fname in batch_files:
            path = os.path.join(TEST_DIR, fname)
            pil_images.append(Image.open(path).convert("RGB"))

        batch_center = torch.stack(
            [transform_centercrop(img) for img in pil_images]
        ).to(DEVICE)

        batch_resize = torch.stack(
            [transform_resize(img) for img in pil_images]
        ).to(DEVICE)

        batch_5crop = torch.stack(
            [transform_fivecrop(img) for img in pil_images]
        ).to(DEVICE)
        bs, n_crops, c, h, w = batch_5crop.shape
        batch_5crop_flat = batch_5crop.view(-1, c, h, w)

        batch_preds = torch.zeros((current_batch_size, NUM_CLASSES)).to(DEVICE)

        for model in models_list:
            out_center = model(batch_center)
            prob_center = F.softmax(out_center, dim=1)

            out_resize = model(batch_resize)
            prob_resize = F.softmax(out_resize, dim=1)

            out_5crop = model(batch_5crop_flat)
            prob_5crop = F.softmax(out_5crop, dim=1)
            prob_5crop = prob_5crop.view(bs, n_crops, NUM_CLASSES).mean(dim=1)

            model_prob = (prob_center + prob_resize + prob_5crop) / 3.0
            batch_preds += model_prob

        batch_preds /= len(models_list)
        final_preds.append(batch_preds.cpu().numpy())

# 6. 결과 출력
final_preds = np.concatenate(final_preds, axis=0)
df_submit = pd.DataFrame(final_preds, columns=[f"c{i}" for i in range(10)])
df_submit.insert(0, "img", img_names)

save_filename = "vgg19_mark26_GAP_tta_ensemble_final.csv"
df_submit.to_csv(save_filename, index=False)
print(f"Done! Submission saved to {save_filename}")
print(df_submit.head())
