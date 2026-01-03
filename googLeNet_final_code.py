# 1. 기본 설정
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import GoogLeNet_Weights
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import StratifiedGroupKFold

EXCEL_PATH      = "/media/strider/54EC1B30EC1B0BBC/deeplearningprojects/data/driver_imgs_list.csv"
SAMPLE_SUB_PATH = "/media/strider/54EC1B30EC1B0BBC/deeplearningprojects/data/sample_submission.csv"
IMG_DIR         = "/media/strider/54EC1B30EC1B0BBC/deeplearningprojects/data/imgs"

IMG_COL   = "img"
LABEL_COL = "classname"

BATCH_SIZE  = 64
NUM_EPOCHS  = 6
LR          = 1e-3
WEIGHT_DECAY = 1e-4
N_FOLDS     = 5

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available!")
device = torch.device("cuda")
print("Using device:", device)

# 2. 데이터 전처리
df = pd.read_csv(EXCEL_PATH)

df["label_id"], class_names = pd.factorize(df[LABEL_COL])
num_classes = len(class_names)

print("클래스 목록:", list(class_names))
print("클래스 개수:", num_classes)

df["driver_id"] = df["subject"]

df["fold"] = -1
sgkf = StratifiedGroupKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=42
)

for fold, (tr_idx, val_idx) in enumerate(
    sgkf.split(df, df["label_id"], groups=df["driver_id"])
):
    df.loc[val_idx, "fold"] = fold

print("Fold 분포:\n", df["fold"].value_counts())

data_dir = os.path.dirname(EXCEL_PATH)
df.to_csv(os.path.join(data_dir, "driver_imgs_folds.csv"), index=False)

## Dataset 정의 
class DriverDataset(Dataset):
    def __init__(self, df, img_dir, img_col, label_col, transform=None):
        """
        df: label_id와 fold가 포함된 DataFrame
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name   = row[self.img_col]
        class_name = row[self.label_col]
        label_id   = row["label_id"]

        img_path = os.path.join(
            self.img_dir,
            "train",
            class_name,
            img_name
        )

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_id


class TestDataset(Dataset):
    def __init__(self, img_dir, img_names, transform=None):
        self.img_dir = img_dir
        self.img_names = img_names
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, "test", img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, img_name


## Transform
train_transform = transforms.Compose([
    transforms.Resize((229, 229)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((229, 229)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])



# 3. 훈련

def build_googlenet(num_classes):
    m = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1,
                         aux_logits=True)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.aux1.fc2 = nn.Linear(m.aux1.fc2.in_features, num_classes)
    m.aux2.fc2 = nn.Linear(m.aux2.fc2.in_features, num_classes)
    return m

fold_best_val_acc = [0.0] * N_FOLDS

for fold in range(N_FOLDS):
    print(f"\n================ FOLD {fold} ================\n")

    df_train = df[df["fold"] != fold].reset_index(drop=True)
    df_val   = df[df["fold"] == fold].reset_index(drop=True)

    train_dataset = DriverDataset(
        df=df_train,
        img_dir=IMG_DIR,
        img_col=IMG_COL,
        label_col=LABEL_COL,
        transform=train_transform
    )

    val_dataset = DriverDataset(
        df=df_val,
        img_dir=IMG_DIR,
        img_col=IMG_COL,
        label_col=LABEL_COL,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6
    )

    ## 모델
    model = build_googlenet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6
    )

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        
        ## TRAIN
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            main_out = outputs.logits
            aux1_out = outputs.aux_logits1
            aux2_out = outputs.aux_logits2

            loss_main = criterion(main_out, labels)
            loss_aux1 = criterion(aux1_out, labels)
            loss_aux2 = criterion(aux2_out, labels)

            loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(main_out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if step % 10 == 0:
                print(f"[Fold {fold}] [Train] Epoch [{epoch+1}/{NUM_EPOCHS}] "
                      f"Step [{step}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        ## VALIDATION
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                logits = outputs

                loss = criterion(logits, labels)
                val_running_loss += loss.item() * images.size(0)

                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc  = correct / total * 100.0
        epoch_val_loss   = val_running_loss / len(val_dataset)
        epoch_val_acc    = val_correct / val_total * 100.0

        print(f"\n==> [Fold {fold}] Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), f"best_googlenet_fold{fold}.pth")
            print(f"*** [Fold {fold}] Best model updated! (Val Acc: {best_val_acc:.2f}%) ***\n")

        scheduler.step()
        print(f"[Fold {fold}] Current LR: {scheduler.get_last_lr()[0]:.6f}")

    fold_best_val_acc[fold] = best_val_acc

print("Fold별 best val acc:", fold_best_val_acc)




# 4. 테스트
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_img_names = sample_sub["img"].tolist()

test_transform = val_transform

test_dataset = TestDataset(
    img_dir=IMG_DIR,
    img_names=test_img_names,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=6
)

print("테스트 이미지 개수:", len(test_dataset))

## 각 fold별 예측
all_fold_probs = []

for fold in range(N_FOLDS):
    print(f"[Predict] Fold {fold} ...")

    model = build_googlenet(num_classes=num_classes).to(device)
    state = torch.load(f"best_googlenet_fold{fold}.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    fold_probs = []

    with torch.no_grad():
        for step, (images, img_names) in enumerate(test_loader, start=1):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            fold_probs.append(probs.cpu().numpy())

    fold_probs = np.concatenate(fold_probs, axis=0)
    all_fold_probs.append(fold_probs)

all_fold_probs = np.stack(all_fold_probs, axis=0)
print("all_fold_probs shape:", all_fold_probs.shape)

## 가중 평균 앙상블 
weights = np.array(fold_best_val_acc, dtype=np.float64)

if weights.sum() == 0:
    weights = np.ones_like(weights) / len(weights)
else:
    weights = weights / weights.sum()

print("앙상블 가중치:", weights)

ensemble_probs = np.tensordot(weights, all_fold_probs, axes=(0, 0))
print("ensemble_probs shape:", ensemble_probs.shape)



# 5. 결과 출력
class_cols = [f"c{i}" for i in range(num_classes)]

submission = pd.DataFrame({"img": test_img_names})
for i, col in enumerate(class_cols):
    submission[col] = ensemble_probs[:, i]

submission = submission[["img"] + class_cols]

SUBMISSION_PATH = "/media/strider/54EC1B30EC1B0BBC/deeplearningprojects/submission_GoogLeNet_5fold_try3.csv"
submission.to_csv(SUBMISSION_PATH, index=False)

print("저장 완료:", SUBMISSION_PATH)