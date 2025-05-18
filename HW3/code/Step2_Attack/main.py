import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# ---------- CONFIG ----------
IMAGE_DIR = "./blurred_images/"
LABEL_FILE = "./identity_CelebA.txt"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------

# ---------- Dataset ----------
class BlurredFaceDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["identity"]
        return image, label

# ---------- Load labels ----------
def load_filtered_identities(label_file, image_dir):
    df = pd.read_csv(label_file, sep=' ', header=None, names=["filename", "identity"])
    available_imgs = set(os.listdir(image_dir))
    df = df[df["filename"].isin(available_imgs)]

    # 原始統計
    original_total = len(df)
    original_classes = df["identity"].nunique()

    # 篩掉出現次數 < 2 的 identity
    valid_ids = df["identity"].value_counts()
    valid_ids = valid_ids[valid_ids >= 10].index
    filtered_df = df[df["identity"].isin(valid_ids)]

    # 被篩掉的統計
    removed_total = original_total - len(filtered_df)
    removed_classes = original_classes - filtered_df["identity"].nunique()

    # 重新編碼 label
    label2id = {label: idx for idx, label in enumerate(sorted(filtered_df["identity"].unique()))}
    filtered_df = filtered_df.copy()
    filtered_df["identity"] = filtered_df["identity"].map(label2id)

    print(f"Original images: {original_total}, classes: {original_classes}")
    print(f"Filtered out:    {removed_total} images, {removed_classes} classes (with < 10 images)")
    print(f"Remaining:       {len(filtered_df)} images, {len(label2id)} classes")

    return filtered_df, len(label2id)


# ---------- Train/Test split ----------
def prepare_datasets(df):
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["identity"], random_state=42)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    train_set = BlurredFaceDataset(train_df, IMAGE_DIR, transform)
    val_set = BlurredFaceDataset(val_df, IMAGE_DIR, transform)
    return train_set, val_set

# ---------- Train ----------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.to(DEVICE)
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()

        acc = total_correct / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {total_loss:.4f} Acc: {acc:.4f}")

        # validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_loader.dataset)
        print(f"           Val Acc: {val_acc:.4f}")

# ---------- Main ----------
def main():
    df, num_classes = load_filtered_identities(LABEL_FILE, IMAGE_DIR)
    print(f"Total images: {len(df)}, classes: {num_classes}")
    
    train_set, val_set = prepare_datasets(df)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # use pretrained resnet18
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

if __name__ == "__main__":
    main()