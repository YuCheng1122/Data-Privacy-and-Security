import argparse
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
CLEAR_DIR = "./celeba_original/"
BLUR_DIR = "./blurred_images_50/"
DP_PIX_DIR = "./dp_pix/"
DP_BLUR_DIR = "./dp_blur/"
LABEL_FILE = "./identity_CelebA.txt"
# MODEL_SAVE_PATH = "./resnet18_dp_pix.pt"
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_SAMPLES_PER_ID = 20
# ----------------------------

class FaceDataset(Dataset):
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
        return image, row["identity"]

def load_and_filter_identities(label_file, img_dir, label2id=None):
    df = pd.read_csv(label_file, sep=' ', header=None, names=["filename", "identity"])
    df = df[df["filename"].isin(os.listdir(img_dir))]

    original_total = len(df)
    original_classes = df["identity"].nunique()

    valid_ids = df["identity"].value_counts()
    valid_ids = valid_ids[valid_ids >= MIN_SAMPLES_PER_ID].index
    filtered_df = df[df["identity"].isin(valid_ids)]

    removed_total = original_total - len(filtered_df)
    removed_classes = original_classes - filtered_df["identity"].nunique()

    if label2id is None:
        label2id = {label: idx for idx, label in enumerate(sorted(filtered_df["identity"].unique()))}
    filtered_df = filtered_df.copy()
    filtered_df["identity"] = filtered_df["identity"].map(label2id)

    print(f"Images from: {img_dir}")
    print(f"Original images: {original_total}, classes: {original_classes}")
    print(f"Filtered out:    {removed_total} images, {removed_classes} classes (< {MIN_SAMPLES_PER_ID} images)")
    print(f"Remaining:       {len(filtered_df)} images, {len(label2id)} classes\n")

    return filtered_df, label2id

def prepare_datasets(train_dir, test_dir):
    train_df, label2id = load_and_filter_identities(LABEL_FILE, train_dir)
    test_df, _ = load_and_filter_identities(LABEL_FILE, test_dir, label2id=label2id)

    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df["identity"], random_state=42)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    train_set = FaceDataset(train_df, train_dir, transform)
    val_set = FaceDataset(val_df, train_dir, transform)
    test_set = FaceDataset(test_df, test_dir, transform)
    return train_set, val_set, test_set, len(label2id)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
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

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_loader.dataset)
        print(f"           Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def test_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            correct += (outputs.argmax(1) == labels).sum().item()
    test_acc = correct / len(test_loader.dataset)
    print(f"           Test Acc: {test_acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', choices=['clear', 'blurred', 'dp_pix', 'dp_blur'], required=True,
                        help="Directory for training images")
    parser.add_argument('--test_dir', choices=['clear', 'blurred', 'dp_pix', 'dp_blur'], required=True,
                        help="Directory for testing images")
    args = parser.parse_args()

    # Generate model save path dynamically based on training directory
    model_save_path = f"./resnet18_{args.train_dir}.pt"

    # Map directory name to actual folder path
    def get_path(name):
        if name == 'clear':
            return CLEAR_DIR
        elif name == 'blurred':
            return BLUR_DIR
        elif name == 'dp_pix':
            return DP_PIX_DIR
        elif name == 'dp_blur':
            return DP_BLUR_DIR
        else:
            raise ValueError(f"Unknown directory alias: {name}")

    train_path = get_path(args.train_dir)
    test_path = get_path(args.test_dir)

    # Prepare dataset loaders
    train_set, val_set, test_set, num_classes = prepare_datasets(train_path, test_path)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize ResNet18 model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train and validate the model
    train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, model_save_path)

    # Test the model
    test_model(model, test_loader)

if __name__ == "__main__":
    main()
